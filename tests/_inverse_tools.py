"""Helpers for the differentiable-inverse-problem tutorial.

The existing gradient evidence in this repository
(``benchmarks/verify_differentiability.py``,
``benchmarks/verify_diffusive_differentiability.py``) answers the question
*"is the autograd gradient equal to the derivative of the discrete scheme?"*.
It does **not** answer the two questions this tutorial is about:

1. Does an autograd gradient actually drive a parameter estimate to the truth
   (i.e. is the gradient usable inside an optimisation loop)?
2. Does the estimate converge to the **continuum** value under mesh
   refinement, i.e. is parameter identification *consistent* and not merely
   self-consistent with the data-generating discretisation?

Both need a scalar-parameter forward model ``theta -> u_T``, a least-squares
misfit, and a reference truth that is independent of the gradient code.  The
helpers here build exactly that, on top of the shipped differentiable entry
points :func:`tensordg.differentiable.advance_advection_1d` (hyperbolic) and
:func:`tensordg.differentiable.advance_heat_1d` (parabolic, SIPG +
Crank--Nicolson).

Numerical convention (inherited from the differentiable entry points): ``dt``
is *frozen*.  A fit therefore returns the parameter of the **discrete** model at
that step size, which is exactly what a central difference of the same code
measures -- see point 2 in the tutorial for why that matters for a CFL-driven
solver.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from benchmarks.verify_differentiability import advection_initial
from tensordg.advection_1d import _operators as _advection_operators
from tensordg.differentiable import advance_advection_1d, advance_heat_1d
from tensordg.heat_1d import _operators as _heat_operators
from tensordg.heat_1d import solve_heat_1d

DTYPE = torch.float64
DEV = torch.device("cpu")


# ---------------------------------------------------------------------------
# L2 projection (the same quadrature rule the solvers use)
# ---------------------------------------------------------------------------

def project_1d(values, ncells: int, degree: int) -> torch.Tensor:
    """L2-project a callable ``values(x) -> tensor`` onto modal Legendre modes.

    Uses ``int_{-1}^{1} P_k P_l dxi = 2 delta_kl / (2k+1)``, i.e. the
    unnormalised basis convention of the solvers.
    """
    h = 1.0 / ncells
    nodes_np, weights_np = np.polynomial.legendre.leggauss(max(2 * degree + 3, 4))
    nodes = torch.as_tensor(nodes_np, dtype=DTYPE)
    weights = torch.as_tensor(weights_np, dtype=DTYPE)
    _, _, phi, _, _, _ = _advection_operators(degree, DEV, DTYPE)
    centers = (torch.arange(ncells, dtype=DTYPE) + 0.5) * h
    xq = centers[:, None] + 0.5 * h * nodes[None, :]
    sampled = values(xq)
    return torch.stack(
        [
            (2 * k + 1) * 0.5 * torch.sum(weights[None, :] * sampled * phi[:, k][None, :], dim=1)
            for k in range(degree + 1)
        ],
        dim=1,
    )


def project_dirichlet_1d(values, ncells: int, degree: int) -> torch.Tensor:
    """Same projection on the SIPG heat mesh (quadrature from ``heat_1d``)."""
    h = 1.0 / ncells
    node_list, weight_list, phi, = _heat_operators(degree, DEV, DTYPE)[:3]
    centers = (torch.arange(ncells, dtype=DTYPE) + 0.5) * h
    xq = centers[:, None] + 0.5 * h * node_list[None, :]
    sampled = values(xq)
    return torch.stack(
        [
            (2 * k + 1) * 0.5 * torch.sum(weight_list[None, :] * sampled * phi[:, k][None, :], dim=1)
            for k in range(degree + 1)
        ],
        dim=1,
    )


# ---------------------------------------------------------------------------
# Case A -- hyperbolic: unknown advection speed (periodic linear advection)
# ---------------------------------------------------------------------------

def advection_setup(ncells: int = 32, degree: int = 2, nsteps: int = 40,
                    cfl: float = 0.2) -> dict:
    """P-degree upwind DG + RK4 setup with a frozen, CFL-consistent ``dt``."""
    h = 1.0 / ncells
    dt = cfl * h / (2 * degree + 1)
    return {
        "ncells": ncells,
        "degree": degree,
        "nsteps": nsteps,
        "h": h,
        "dt": dt,
        "final_time": nsteps * dt,
        "coeff0": advection_initial(ncells, degree),
    }


def advection_field(setup: dict, speed):
    """Terminal modal field ``u_T`` for a (possibly differentiable) ``speed``."""
    return advance_advection_1d(
        setup["coeff0"], setup["ncells"], setup["degree"], speed=speed,
        nsteps=setup["nsteps"], dt=setup["dt"],
    )


def advection_data(setup: dict, speed=1.0) -> torch.Tensor:
    """Synthetic "measurement": the terminal field of the *discrete* model."""
    return advection_field(setup, speed).detach()


def advection_continuum_data(setup: dict, speed=1.0) -> torch.Tensor:
    """Terminal field of the *continuum* solution ``sin(2 pi (x - c t))``."""
    shift = float(speed) * setup["final_time"]
    return project_1d(
        lambda x: torch.sin(2.0 * math.pi * (x - shift)),
        setup["ncells"], setup["degree"],
    )


def least_squares_misfit(field, data: torch.Tensor):
    """``0.5 * ||field(theta) - data||_F^2`` -- a plain differentiable functional."""

    def objective(theta):
        residual = field(theta) - data
        return 0.5 * torch.sum(residual * residual)

    return objective


def advection_objective(setup: dict, data: torch.Tensor):
    return least_squares_misfit(lambda theta: advection_field(setup, theta), data)


# ---------------------------------------------------------------------------
# Case B -- parabolic: unknown diffusivity (SIPG heat, Crank--Nicolson)
# ---------------------------------------------------------------------------

def heat_setup(ncells: int = 24, degree: int = 2, nsteps: int = 20,
               final_time: float = 0.02, penalty: float = 20.0) -> dict:
    """Zero-Dirichlet SIPG heat setup; initial data from the shipped solver."""
    _, coeff0, _ = solve_heat_1d(ncells=ncells, degree=degree, final_time=0.0)
    return {
        "ncells": ncells,
        "degree": degree,
        "nsteps": nsteps,
        "final_time": final_time,
        "dt": final_time / nsteps,
        "penalty": penalty,
        "h": 1.0 / ncells,
        "coeff0": coeff0,
    }


def heat_field(setup: dict, alpha):
    return advance_heat_1d(
        setup["coeff0"], setup["ncells"], setup["degree"], nsteps=setup["nsteps"],
        dt=setup["dt"], alpha=alpha, penalty=setup["penalty"],
    )


def heat_data(setup: dict, alpha=0.2) -> torch.Tensor:
    """Measurement from the discrete model at the true diffusivity."""
    return heat_field(setup, torch.as_tensor(alpha, dtype=DTYPE)).detach()


def heat_continuum_data(setup: dict, alpha: float = 0.2) -> torch.Tensor:
    """Measurement from the continuum solution ``sin(pi x) exp(-pi^2 alpha t)``."""
    decay = math.exp(-math.pi**2 * alpha * setup["final_time"])
    return project_dirichlet_1d(
        lambda x: torch.sin(math.pi * x) * decay, setup["ncells"], setup["degree"],
    )


def heat_objective(setup: dict, data: torch.Tensor):
    return least_squares_misfit(lambda alpha: heat_field(setup, alpha), data)


# ---------------------------------------------------------------------------
# Scalar differentiation / minimisation utilities
# ---------------------------------------------------------------------------

def value_and_grad(objective, theta_value: float) -> tuple[float, float]:
    """Objective value and **autograd** derivative at a scalar parameter."""
    theta = torch.tensor(float(theta_value), dtype=DTYPE, requires_grad=True)
    value = objective(theta)
    (grad,) = torch.autograd.grad(value, theta)
    return float(value), float(grad)


def central_difference(objective, theta_value: float, eps: float = 1.0e-6) -> float:
    """Independent reference derivative: central difference of the same code."""
    plus = float(objective(torch.tensor(float(theta_value) + eps, dtype=DTYPE)))
    minus = float(objective(torch.tensor(float(theta_value) - eps, dtype=DTYPE)))
    return (plus - minus) / (2.0 * eps)


def relative_error(estimate: float, truth: float) -> float:
    return abs(float(estimate) - float(truth)) / max(abs(float(truth)), 1.0e-300)


def scan(objective, values) -> list[dict]:
    """Evaluate the misfit on a parameter grid (for landscape figures/tests)."""
    return [
        {"theta": float(v), "objective": float(objective(torch.tensor(float(v), dtype=DTYPE)))}
        for v in values
    ]


def secant_fit(objective, theta0: float, bounds=None, max_iter: int = 30,
               tol: float = 1.0e-12, bump: float = 0.05) -> dict:
    """1-D minimisation driven **only** by autograd gradients (secant/Newton).

    ``theta_{n+1} = theta_n - g_n * (theta_n - theta_{n-1}) / (g_n - g_{n-1})``.
    For a smooth misfit this converges superlinearly in a handful of iterations,
    which is the cheapest honest demonstration that the gradient is a usable
    descent direction.  ``bounds`` clamps iterates into a physical interval
    (e.g. ``speed > 0``).
    """
    theta_prev = float(theta0)
    theta_cur = float(theta0) * (1.0 + bump)
    grad_prev = value_and_grad(objective, theta_prev)[1]

    def clamp(value: float) -> float:
        if bounds is None:
            return value
        return min(max(value, bounds[0]), bounds[1])

    theta_cur = clamp(theta_cur)
    history: list[dict] = []
    for iteration in range(max_iter):
        value, grad = value_and_grad(objective, theta_cur)
        history.append(
            {"iteration": iteration, "theta": theta_cur, "objective": value, "gradient": grad}
        )
        if abs(grad) <= tol:
            break
        denominator = grad - grad_prev
        if abs(denominator) <= 1.0e-300:
            break
        theta_next = clamp(theta_cur - grad * (theta_cur - theta_prev) / denominator)
        if abs(theta_next - theta_cur) <= 1.0e-14 * max(1.0, abs(theta_cur)):
            theta_cur = theta_next
            break
        theta_prev, grad_prev, theta_cur = theta_cur, grad, theta_next
    value, grad = value_and_grad(objective, theta_cur)
    return {
        "theta": theta_cur,
        "objective": value,
        "gradient": grad,
        "iterations": len(history),
        "history": history,
    }


def add_relative_noise(field: torch.Tensor, level: float, seed: int = 20261103) -> torch.Tensor:
    """Multiplicative Gaussian noise with standard deviation ``level``."""
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(field.shape, dtype=field.dtype, generator=generator)
    return field * (1.0 + level * noise)