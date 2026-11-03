"""Regression tests for autograd sensitivities of the diffusive (SIPG + CN) path.

The three references are independent (see
``benchmarks/verify_diffusive_differentiability.py``):

* the shipped :func:`tensordg.heat_1d.solve_heat_1d` (discrete-scheme identity);
* a hand-assembled exact discrete adjoint (no autograd, no finite differences);
* the analytical mode-decay sensitivity of ``u_t = alpha u_xx``.
"""
from __future__ import annotations

import math

import pytest
import torch

from tensordg.differentiable import (
    advance_heat_1d,
    heat_1d_operators,
    legendre_l2_sq,
    relative_difference,
)
from tensordg.heat_1d import solve_heat_1d

FINAL_TIME = 0.1
PENALTY = 20.0


def _steps_dt(ncells: int, final_time: float = FINAL_TIME) -> tuple[int, float]:
    steps = max(1, math.ceil(final_time / (0.02 * (1.0 / ncells))))
    return steps, final_time / steps


def _initial(ncells: int, degree: int) -> torch.Tensor:
    _centers, coeff0, _meta = solve_heat_1d(
        ncells=ncells, degree=degree, final_time=0.0, penalty=PENALTY
    )
    return coeff0


def _exact_adjoint(coeff0: torch.Tensor, ncells: int, degree: int, steps: int,
                   dt: float, alpha: float) -> tuple[float, torch.Tensor]:
    """Exact adjoint of ``u_{n+1} = G u_n`` with ``G = A^{-1} B`` (CN + SIPG)."""
    mass, diffusion, _centers = heat_1d_operators(ncells, degree, PENALTY)
    identity = torch.eye(mass.shape[0], dtype=mass.dtype)
    system = mass + 0.5 * dt * alpha * diffusion
    explicit = mass - 0.5 * dt * alpha * diffusion
    step_map = torch.linalg.solve(system, explicit @ identity)
    d_step_map = torch.linalg.solve(
        system, -0.5 * dt * diffusion - 0.5 * dt * diffusion @ step_map
    )
    states = [coeff0.reshape(-1)]
    for _ in range(steps):
        states.append(step_map @ states[-1])
    adjoint = 2.0 * (mass @ states[-1])
    d_alpha = 0.0
    for index in range(steps, 0, -1):
        d_alpha += float(adjoint @ (d_step_map @ states[index - 1]))
        adjoint = step_map.T @ adjoint
    return d_alpha, adjoint.reshape(coeff0.shape)


@pytest.mark.parametrize("degree", [1, 2, 3])
def test_advance_matches_shipped_heat_solver(degree):
    """``advance_heat_1d(alpha=1)`` must reproduce ``solve_heat_1d`` bit for bit."""
    ncells = 8
    coeff0 = _initial(ncells, degree)
    _c, reference, meta = solve_heat_1d(
        ncells=ncells, degree=degree, final_time=FINAL_TIME,
        penalty=PENALTY,
    )
    mirrored = advance_heat_1d(
        coeff0, ncells, degree, nsteps=meta["steps"], dt=meta["dt"],
        alpha=1.0, penalty=PENALTY,
    )
    assert float((mirrored - reference).abs().max()) < 1e-12


@pytest.mark.parametrize("ncells,degree", [(8, 2), (16, 1)])
def test_autograd_alpha_matches_exact_discrete_adjoint(ncells, degree):
    """``dJ/dalpha`` from autograd vs the hand-assembled adjoint (machine precision)."""
    alpha = 0.05
    coeff0 = _initial(ncells, degree)
    steps, dt = _steps_dt(ncells)
    alpha_t = torch.tensor(alpha, dtype=torch.float64, requires_grad=True)
    final = advance_heat_1d(
        coeff0, ncells, degree, nsteps=steps, dt=dt, alpha=alpha_t, penalty=PENALTY
    )
    value = legendre_l2_sq(final, 1.0 / ncells)
    (grad_alpha,) = torch.autograd.grad(value, alpha_t)
    reference, _grad_state = _exact_adjoint(coeff0, ncells, degree, steps, dt, alpha)
    relative = abs(float(grad_alpha) - reference) / abs(reference)
    assert relative < 1e-12, relative


def test_autograd_initial_state_matches_exact_discrete_adjoint():
    """``dJ/dcoeff0`` (all modal dofs) vs the exact adjoint gradient."""
    ncells, degree, alpha = 8, 2, 0.05
    coeff0 = _initial(ncells, degree)
    steps, dt = _steps_dt(ncells)
    state0 = coeff0.clone().requires_grad_(True)
    final = advance_heat_1d(
        state0, ncells, degree, nsteps=steps, dt=dt, alpha=alpha, penalty=PENALTY
    )
    (grad_state,) = torch.autograd.grad(legendre_l2_sq(final, 1.0 / ncells), state0)
    _d_alpha, reference = _exact_adjoint(coeff0, ncells, degree, steps, dt, alpha)
    assert relative_difference(grad_state, reference) < 1e-10


def test_autograd_matches_central_difference_in_alpha():
    """Autograd vs central difference of the *same* discrete scheme."""
    ncells, degree, alpha, eps = 8, 2, 0.05, 1e-5
    coeff0 = _initial(ncells, degree)
    steps, dt = _steps_dt(ncells)
    alpha_t = torch.tensor(alpha, dtype=torch.float64, requires_grad=True)
    final = advance_heat_1d(
        coeff0, ncells, degree, nsteps=steps, dt=dt, alpha=alpha_t, penalty=PENALTY
    )
    (grad_alpha,) = torch.autograd.grad(legendre_l2_sq(final, 1.0 / ncells), alpha_t)
    values = []
    for sign in (1.0, -1.0):
        with torch.no_grad():
            shifted = advance_heat_1d(
                coeff0, ncells, degree, nsteps=steps, dt=dt,
                alpha=alpha + sign * eps, penalty=PENALTY,
            )
            values.append(float(legendre_l2_sq(shifted, 1.0 / ncells)))
    finite_difference = (values[0] - values[1]) / (2.0 * eps)
    relative = abs(float(grad_alpha) - finite_difference) / abs(finite_difference)
    assert relative < 1e-7, relative


def test_sensitivity_matches_physics_within_three_percent():
    """Sensitivity tangent vs analytical mode decay, and its convergence with p/h.

    ``J(alpha) = exp(-2 pi^2 alpha T)/2`` and ``dJ/dalpha = -2 pi^2 T J``.
    The residual gap is pure discretisation error, so it must shrink as the
    polynomial degree grows: 3% gate at P1/8 cells, then six orders tighter.
    """
    alpha = 0.05
    analytic = -2.0 * math.pi ** 2 * FINAL_TIME * 0.5 * math.exp(
        -2.0 * math.pi ** 2 * alpha * FINAL_TIME
    )
    errors = {}
    for ncells, degree in ((8, 1), (8, 2), (16, 2), (16, 3)):
        coeff0 = _initial(ncells, degree)
        steps, dt = _steps_dt(ncells)
        alpha_t = torch.tensor(alpha, dtype=torch.float64, requires_grad=True)
        final = advance_heat_1d(
            coeff0, ncells, degree, nsteps=steps, dt=dt, alpha=alpha_t, penalty=PENALTY
        )
        (grad_alpha,) = torch.autograd.grad(legendre_l2_sq(final, 1.0 / ncells), alpha_t)
        errors[(ncells, degree)] = abs(float(grad_alpha) - analytic) / abs(analytic)
    assert errors[(8, 1)] < 0.03
    assert errors[(8, 2)] < 1e-4
    assert errors[(16, 2)] < 1e-5
    assert errors[(16, 3)] < 1e-6
    # monotone improvement with degree at fixed resolution
    assert errors[(8, 2)] < errors[(8, 1)] / 100.0


def test_objective_is_second_order_accurate_in_energy():
    """The discrete energy functional itself stays inside the 3% documentation gate."""
    alpha = 0.05
    analytic = 0.5 * math.exp(-2.0 * math.pi ** 2 * alpha * FINAL_TIME)
    for ncells, degree in ((8, 1), (16, 2)):
        coeff0 = _initial(ncells, degree)
        steps, dt = _steps_dt(ncells)
        with torch.no_grad():
            final = advance_heat_1d(
                coeff0, ncells, degree, nsteps=steps, dt=dt, alpha=alpha,
                penalty=PENALTY,
            )
            value = float(legendre_l2_sq(final, 1.0 / ncells))
        assert abs(value - analytic) / analytic < 0.03
    assert coeff0.dtype == torch.float64