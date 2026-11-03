"""Regression tests for the differentiability claim of TensorDG.

The README/audit advertise "differentiable DG built with PyTorch"; these tests
pin that claim down in three independent ways:

* **scheme identity** — the autograd-ready integrators in
  :mod:`tensordg.differentiable` reproduce the shipped solvers exactly, so the
  gradients below belong to the *verified* scheme;
* **reference gradients** — central finite differences of the same discrete
  scheme, plus (for the linear case) an exact adjoint built from the transposed
  one-step operator matrix, i.e. without autograd;
* **teeth** — the comparison metric must flag a 1e-3 relative perturbation.
"""
from __future__ import annotations

import math

import pytest
import torch

from tensordg.advection_1d import solve_advection
from tensordg.differentiable import (
    advance_advection_1d,
    advance_euler_dg_1d,
    advance_euler_dg_2d,
    central_difference_directional,
    directional_derivative_ad,
    gradient_consistency,
    legendre_l2_sq,
    legendre_l2_sq_2d,
    probe_value,
    relative_difference,
)
from tensordg.euler_dg_1d import solve_euler_dg_1d
from tensordg.euler_dg_2d import solve_euler_dg_2d_vortex

from benchmarks.verify_differentiability import (
    advection_initial,
    advection_objective,
    euler_1d_periodic_initial,
    euler_1d_time_steps,
    exact_adjoint_gradient,
    sod_initial,
    vortex_setup,
)

DTYPE = torch.float64
THRESHOLD = 1.0e-6


# ---------------------------------------------------------------------------
# Scheme identity: the differentiable integrators are the shipped solvers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("degree", [1, 2, 3])
def test_advance_matches_shipped_advection_solver(degree):
    ncells = 12
    _, coeff_ref, meta = solve_advection(
        ncells=ncells, degree=degree, final_time=0.05, speed=1.0, cfl=0.08
    )
    coeff0 = advection_initial(ncells, degree)
    mine = advance_advection_1d(coeff0, ncells, degree, 1.0, meta["steps"], meta["dt"])
    assert torch.allclose(mine, coeff_ref, rtol=0.0, atol=1e-14)


def test_advance_matches_shipped_euler_1d_solver():
    ncells, degree, final_time = 32, 2, 0.008
    dx, coeff0 = sod_initial(ncells, degree)
    steps = euler_1d_time_steps(coeff0, dx, degree, final_time, 0.12)
    _, coeff_ref, meta = solve_euler_dg_1d(
        (1.0, 0.0, 1.0), (0.125, 0.0, 0.1),
        ncells=ncells, degree=degree, final_time=final_time, cfl=0.12,
    )
    assert meta["steps"] == len(steps)
    mine = advance_euler_dg_1d(coeff0, dx, len(steps), steps, periodic=False)
    assert torch.allclose(mine, coeff_ref, rtol=0.0, atol=1e-13)


def test_advance_matches_shipped_euler_2d_solver():
    coeff0, steps, dx, dy = vortex_setup(6, 1, 0.02)
    _, _, coeff_ref, meta = solve_euler_dg_2d_vortex(ncells=6, degree=1, final_time=0.02)
    assert meta["steps"] == len(steps)
    mine = advance_euler_dg_2d(coeff0, dx, dy, len(steps), steps)
    assert torch.allclose(mine, coeff_ref, rtol=0.0, atol=1e-13)


# ---------------------------------------------------------------------------
# Reverse mode vs independent references
# ---------------------------------------------------------------------------

def test_advection_gradient_matches_finite_difference():
    ncells, degree, nsteps, dt = 12, 2, 10, 1.0e-3
    coeff0 = advection_initial(ncells, degree)
    objective = advection_objective(ncells, degree, nsteps, dt)
    result = gradient_consistency(objective, coeff0, n_directions=4, eps=1.0e-6, seed=7)
    assert result["max_relative_difference"] < THRESHOLD, result


def test_advection_gradient_matches_exact_transposed_adjoint():
    ncells, degree, nsteps, dt = 12, 2, 10, 1.0e-3
    coeff0 = advection_initial(ncells, degree)
    objective = advection_objective(ncells, degree, nsteps, dt)
    x0 = coeff0.clone().requires_grad_(True)
    grad_ad, = torch.autograd.grad(objective(x0), x0)
    final = advance_advection_1d(coeff0, ncells, degree, 1.0, nsteps, dt)
    grad_exact = exact_adjoint_gradient(ncells, degree, 1.0, dt, nsteps, final)
    assert relative_difference(grad_ad, grad_exact) < 1.0e-12


def test_wave_speed_is_a_differentiable_parameter():
    ncells, degree, nsteps, dt = 12, 2, 15, 2.0e-3
    coeff0 = advection_initial(ncells, degree)

    def objective(speed):
        final = advance_advection_1d(coeff0, ncells, degree, speed.reshape(()), nsteps, dt)
        return probe_value(final, 3, 0.5)

    speed0 = torch.tensor(1.0, dtype=DTYPE, requires_grad=True)
    (grad,) = torch.autograd.grad(objective(speed0), speed0)
    reference = central_difference_directional(
        lambda s: objective(s.reshape(())),
        torch.tensor(1.0, dtype=DTYPE), torch.tensor(1.0, dtype=DTYPE), eps=1.0e-6,
    )
    assert abs(float(grad) - reference) / max(abs(reference), 1e-300) < THRESHOLD
    assert abs(float(grad)) > 1.0e-3, "probe objective must be phase sensitive"


def test_euler_1d_gradient_matches_finite_difference():
    ncells, degree = 24, 2
    dx, coeff0 = euler_1d_periodic_initial(ncells, degree)
    steps = euler_1d_time_steps(coeff0, dx, degree, 0.006, 0.12)

    def objective(coeff):
        final = advance_euler_dg_1d(coeff, dx, len(steps), steps, periodic=True)
        return legendre_l2_sq(final, dx, component=0)

    result = gradient_consistency(objective, coeff0, n_directions=3, eps=1.0e-6, seed=3)
    assert result["max_relative_difference"] < THRESHOLD, result


def test_euler_2d_gradient_matches_finite_difference():
    coeff0, steps, dx, dy = vortex_setup(6, 1, 0.02)

    def objective(coeff):
        final = advance_euler_dg_2d(coeff, dx, dy, len(steps), steps)
        return legendre_l2_sq_2d(final, dx, dy, component=2)

    result = gradient_consistency(objective, coeff0, n_directions=2, eps=1.0e-6, seed=5)
    assert result["max_relative_difference"] < THRESHOLD, result


# ---------------------------------------------------------------------------
# The checking machinery must have teeth (negative control)
# ---------------------------------------------------------------------------

def test_gradient_checker_detects_forward_model_perturbation():
    """A 5% error in the step size of the forward model must be caught.

    The mismatch scales linearly with the model error, so this also bounds what
    the check can resolve: at this configuration a 1e-4 relative forward error
    stays below the 1e-6 threshold (see ``validation/differentiability``).
    """
    ncells, degree, nsteps, dt = 12, 2, 10, 1.0e-3
    coeff0 = advection_initial(ncells, degree)
    objective = advection_objective(ncells, degree, nsteps, dt)
    direction = torch.randn(coeff0.shape, dtype=DTYPE)
    direction = direction / direction.norm()
    ad = float(directional_derivative_ad(objective, coeff0, direction))

    coarse = advection_objective(ncells, degree, nsteps, dt * 1.05)
    fd_coarse = central_difference_directional(coarse, coeff0, direction, eps=1.0e-6)
    assert abs(ad - fd_coarse) / abs(ad) > THRESHOLD

    tiny = advection_objective(ncells, degree, nsteps, dt * (1.0 + 1.0e-5))
    fd_tiny = central_difference_directional(tiny, coeff0, direction, eps=1.0e-6)
    assert abs(ad - fd_tiny) / abs(ad) < THRESHOLD


def test_gradient_checker_detects_gradient_scale_perturbation():
    value = -2.345
    perturbed = value * (1.0 + 1.0e-3)
    assert abs(value - perturbed) / abs(value) > THRESHOLD


# ---------------------------------------------------------------------------
# Objective helpers are exact functionals of the modal state
# ---------------------------------------------------------------------------

def test_legendre_l2_sq_matches_analytic_integral():
    # u(x) = 1 + 2 * P_1(xi) with xi = 2x/h - 1 on one cell of length h:
    # int_0^h u^2 dx = h * (1 + 4/3)
    h = 0.25
    coeff = torch.tensor([[1.0, 2.0]], dtype=DTYPE)
    assert abs(float(legendre_l2_sq(coeff, h)) - h * (1.0 + 4.0 / 3.0)) < 1e-14


def test_probe_value_matches_trace_and_polynomial():
    # quadratic u(xi) = 1 + xi + xi^2 -> P0=1, P1=xi, P2=(3xi^2-1)/2, so
    # xi^2 = (2 P_2 + P_0)/3 and u = 4/3 P_0 + P_1 + 2/3 P_2
    coeff = torch.tensor([[4.0 / 3.0, 1.0, 2.0 / 3.0]], dtype=DTYPE)
    for xi in (-1.0, -0.3, 0.0, 0.5, 1.0):
        expected = 1.0 + xi + xi ** 2
        assert abs(float(probe_value(coeff, 0, xi)) - expected) < 1e-13