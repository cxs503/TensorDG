"""Autograd-ready DG time integration and gradient-verification utilities.

Motivation
----------
TensorDG advertises "differentiable DG built with PyTorch" as its unique
ecosystem slot, but the public solvers (``solve_advection``,
``solve_euler_dg_1d``, ``solve_euler_dg_2d``) assemble their initial state from
Python floats.  There is therefore *no supported way* to obtain
``d(objective)/d(initial state)`` or ``d(objective)/d(parameter)`` through a
full DG time integration, and nothing in the test suite checked that the
autograd derivatives are actually correct.

This module closes that gap:

1. ``advance_*`` — the same spatial operators and the same RK/SSP-RK
   integrators used by the shipped solvers, exposed as pure functions of a
   modal-coefficient tensor, so the whole time integration stays inside one
   autograd graph.  ``speed`` may be a ``torch`` scalar, i.e. a physical
   parameter can be differentiated through.

2. Objective helpers (``legendre_l2_sq``, ``legendre_l2_sq_2d``,
   ``probe_value``) that are exact functionals of the modal state.

3. Finite-difference reference utilities (``central_difference_*``,
   ``directional_derivative_ad``, ``gradient_consistency``) used by
   ``benchmarks/verify_differentiability.py`` and
   ``tests/test_differentiability.py`` to prove autograd derivatives agree with
   central differences of the *same* discrete scheme.

Convention: the first modal coefficient ``coeff[..., 0, ...]`` is the cell
average (``P_0 = 1``), and Legendre modes are *not* normalised, so
``int_{-1}^{1} P_k P_l dxi = 2 delta_kl / (2k+1)``.
"""
from __future__ import annotations

import torch

from .advection_1d import _operators as _advection_operators
from .euler_dg_1d import _operators as _euler_dg_1d_operators
from .euler_dg_1d import _positivity_filter as _euler_dg_1d_positivity
from .euler_dg_1d import _rhs as _euler_dg_1d_rhs
from .euler_dg_2d import _positivity_filter as _euler_dg_2d_positivity
from .euler_dg_2d import _rhs as _euler_dg_2d_rhs
from .euler_dg_2d import apply_tvb_limiter_2d, legendre_operators
from .limiters import apply_limiter

__all__ = [
    "advance_advection_1d",
    "advance_euler_dg_1d",
    "advance_euler_dg_2d",
    "legendre_l2_sq",
    "legendre_l2_sq_2d",
    "probe_value",
    "directional_derivative_ad",
    "central_difference_directional",
    "central_difference_gradient",
    "gradient_consistency",
    "relative_difference",
]


# ---------------------------------------------------------------------------
# Spatial / temporal operators (differentiable)
# ---------------------------------------------------------------------------

def _advection_rhs(coeff: torch.Tensor, h: float, speed, volume: torch.Tensor,
                   phi_left: torch.Tensor, phi_right: torch.Tensor) -> torch.Tensor:
    """Upwind linear-advection DG residual for a scalar **or tensor** speed.

    For ``speed > 0`` the interface flux at a face is the trace from the left
    cell, for ``speed < 0`` the trace from the right cell.  This is exactly the
    branch taken by :func:`tensordg.advection_1d._rhs`, written with
    ``torch.where`` so that ``speed`` can carry a gradient.
    """
    if not torch.is_tensor(speed):
        speed = torch.as_tensor(speed, dtype=coeff.dtype, device=coeff.device)
    right_trace = coeff @ phi_right
    left_trace = coeff @ phi_left
    positive = speed >= 0
    flux_right = torch.where(positive, right_trace, torch.roll(left_trace, -1, dims=0))
    flux_left = torch.where(positive, torch.roll(right_trace, 1, dims=0), left_trace)
    modes = torch.arange(coeff.shape[1], device=coeff.device, dtype=coeff.dtype)
    scale = ((2.0 * modes + 1.0) / h).unsqueeze(0)
    volume_term = coeff @ volume.T          # [cells, modes] = int P_k du/dxi
    return scale * speed * (
        volume_term
        - flux_right[:, None] * phi_right[None, :]
        + flux_left[:, None] * phi_left[None, :]
    )


def _step_sizes(dt, nsteps: int) -> list[float]:
    """Expand ``dt`` (scalar or per-step sequence) into ``nsteps`` step sizes."""
    if isinstance(dt, (list, tuple)):
        if len(dt) != nsteps:
            raise ValueError("dt sequence must have exactly nsteps entries")
        return [float(value) for value in dt]
    return [float(dt)] * nsteps


def advance_advection_1d(coeff0: torch.Tensor, ncells: int, degree: int,
                         speed=1.0, nsteps: int = 1, dt: float = 0.0) -> torch.Tensor:
    """Classical RK4 integration of ``u_t + speed*u_x = 0`` (periodic, upwind).

    ``coeff0`` is ``[ncells, degree+1]``; ``speed`` may be a 0-dim tensor so
    that ``dJ/dspeed`` is available.  ``dt`` is *frozen* (passed in) — the
    derivative returned by autograd is the tangent of the discrete scheme at
    that fixed step size, which is what the finite-difference reference in
    :func:`central_difference_gradient` also uses.
    """
    if coeff0.dim() != 2 or coeff0.shape[0] != ncells or coeff0.shape[1] != degree + 1:
        raise ValueError("coeff0 must have shape [ncells, degree+1]")
    h = 1.0 / ncells
    _, _, _, volume, phi_left, phi_right = _advection_operators(
        degree, coeff0.device, coeff0.dtype
    )
    arg = (h, speed, volume, phi_left, phi_right)
    coeff = coeff0
    for step in _step_sizes(dt, nsteps):
        k1 = _advection_rhs(coeff, *arg)
        k2 = _advection_rhs(coeff + 0.5 * step * k1, *arg)
        k3 = _advection_rhs(coeff + 0.5 * step * k2, *arg)
        k4 = _advection_rhs(coeff + step * k3, *arg)
        coeff = coeff + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return coeff


def advance_euler_dg_1d(coeff0: torch.Tensor, dx: float, nsteps: int, dt: float,
                        limiter: str | None = None, tvb_constant: float = 1.0,
                        limiter_stage: str = "every",
                        periodic: bool = False) -> torch.Tensor:
    """SSP-RK3 integration of the 1-D Euler DG operator (HLLC + positivity).

    Mirrors the time loop of :func:`tensordg.euler_dg_1d.solve_euler_dg_1d`
    line by line, with ``dt`` frozen so that the mapping
    ``coeff0 -> coeff_final`` is differentiable.  The limiters are
    non-smooth (minmod / branch tests), so gradient checks use
    ``limiter=None``; ``limiter="tvb"`` is still accepted for forward runs.
    """
    if coeff0.dim() != 3 or coeff0.shape[2] != 3:
        raise ValueError("coeff0 must have shape [ncells, degree+1, 3]")
    if limiter_stage not in ("every", "final"):
        raise ValueError("limiter_stage must be 'every' or 'final'")
    ncells, degree = coeff0.shape[0], coeff0.shape[1] - 1
    nodes, weights, phi, dphi, phi_left, phi_right = _euler_dg_1d_operators(
        degree, coeff0.device, coeff0.dtype
    )
    check_points = torch.cat((phi, phi_left[None, :], phi_right[None, :]), dim=0)
    limit_each_stage = limiter_stage == "every"
    coeff = coeff0
    for step in _step_sizes(dt, nsteps):
        k1 = _euler_dg_1d_rhs(coeff, dx, weights, phi, dphi, phi_left, phi_right, periodic)
        u1 = coeff + step * k1
        if limit_each_stage:
            u1 = apply_limiter(u1, dx, kind=limiter, tvb_constant=tvb_constant, periodic=periodic)
        u1 = _euler_dg_1d_positivity(u1, check_points)
        k2 = _euler_dg_1d_rhs(u1, dx, weights, phi, dphi, phi_left, phi_right, periodic)
        u2 = 0.75 * coeff + 0.25 * (u1 + step * k2)
        if limit_each_stage:
            u2 = apply_limiter(u2, dx, kind=limiter, tvb_constant=tvb_constant, periodic=periodic)
        u2 = _euler_dg_1d_positivity(u2, check_points)
        k3 = _euler_dg_1d_rhs(u2, dx, weights, phi, dphi, phi_left, phi_right, periodic)
        coeff = apply_limiter(
            (1.0 / 3.0) * coeff + (2.0 / 3.0) * (u2 + step * k3),
            dx, kind=limiter, tvb_constant=tvb_constant, periodic=periodic,
        )
        coeff = _euler_dg_1d_positivity(coeff, check_points)
    return coeff


def advance_euler_dg_2d(coeff0: torch.Tensor, dx: float, dy: float, nsteps: int,
                        dt: float, quadrature: int | None = None,
                        periodic=(True, True), limiter: str | None = None,
                        tvb_constant: float = 1.0) -> torch.Tensor:
    """SSP-RK3 integration of the 2-D Euler DG operator (HLLC + positivity).

    ``coeff0`` is ``[nx, ny, degree+1, degree+1, 4]``.  Identical to the time
    loop of :func:`tensordg.euler_dg_2d.solve_euler_dg_2d` with ``dt`` frozen.
    """
    if coeff0.dim() != 5 or coeff0.shape[4] != 4:
        raise ValueError("coeff0 must have shape [nx, ny, degree+1, degree+1, 4]")
    degree = coeff0.shape[2] - 1
    order = quadrature if quadrature is not None else max(degree + 2, 3)
    _, weights, basis, derivative, signs = legendre_operators(
        degree, order, coeff0.device, coeff0.dtype
    )

    def impose(state):
        if limiter == "tvb":
            state = apply_tvb_limiter_2d(state, dx, dy, tvb_constant)
        elif limiter is not None:
            raise ValueError("limiter must be None or 'tvb'")
        return _euler_dg_2d_positivity(state, basis, signs)

    coeff = coeff0
    for step in _step_sizes(dt, nsteps):
        k1 = _euler_dg_2d_rhs(coeff, basis, derivative, weights, signs, dx, dy, periodic)
        u1 = impose(coeff + step * k1)
        k2 = _euler_dg_2d_rhs(u1, basis, derivative, weights, signs, dx, dy, periodic)
        u2 = impose(0.75 * coeff + 0.25 * (u1 + step * k2))
        k3 = _euler_dg_2d_rhs(u2, basis, derivative, weights, signs, dx, dy, periodic)
        coeff = impose(coeff / 3.0 + (2.0 / 3.0) * (u2 + step * k3))
    return coeff


# ---------------------------------------------------------------------------
# Differentiable objectives
# ---------------------------------------------------------------------------

def legendre_l2_sq(coeff: torch.Tensor, h: float, component=None) -> torch.Tensor:
    """Exact squared L2 norm of the modal representation on ``ncells`` cells.

    ``int_0^1 u_h^2 dx = h * sum_k c_k^2 / (2k+1)`` because the Legendre modes
    are orthogonal; no quadrature is involved, so the functional is exact for
    the discrete state.
    """
    ncells, nmodes = coeff.shape[0], coeff.shape[1]
    weights = 1.0 / torch.arange(1, 2 * nmodes, 2, device=coeff.device, dtype=coeff.dtype)
    data = coeff if component is None else coeff[..., component]
    return h * (data ** 2 * weights.view(*([1] * (data.dim() - 1)), nmodes)).sum()


def legendre_l2_sq_2d(coeff: torch.Tensor, dx: float, dy: float, component=None) -> torch.Tensor:
    """2-D counterpart of :func:`legendre_l2_sq` (tensor-product modes)."""
    nm = coeff.shape[2]
    w1 = 1.0 / torch.arange(1, 2 * nm, 2, device=coeff.device, dtype=coeff.dtype)
    data = coeff if component is None else coeff[..., component]
    return dx * dy * torch.einsum("xykl,xykl,k,l->", data, data, w1, w1)


def probe_value(coeff: torch.Tensor, cell: int, xi: float, degree: int | None = None) -> torch.Tensor:
    """Point value of the 1-D DG polynomial: ``u_h(xi) = sum_m c_m P_m(xi)``.

    ``xi`` is the reference coordinate in ``[-1, 1]``; a point evaluation is a
    natural differentiable output functional (adjoint sensitivity of a sensor).
    """
    degree = coeff.shape[1] - 1 if degree is None else degree
    xi_t = torch.as_tensor(xi, device=coeff.device, dtype=coeff.dtype)
    # Bonnet recurrence (differentiable, no scipy/torch.special dependency):
    # (k+1) P_{k+1} = (2k+1) xi P_k - k P_{k-1}
    basis = [torch.ones_like(xi_t)]
    if degree >= 1:
        basis.append(xi_t)
    for k in range(1, degree):
        basis.append(((2 * k + 1) * xi_t * basis[k] - k * basis[k - 1]) / (k + 1))
    stacked = torch.stack(basis[: degree + 1])
    return (coeff[cell] * stacked).sum()


# ---------------------------------------------------------------------------
# Finite-difference reference utilities
# ---------------------------------------------------------------------------

def relative_difference(a: torch.Tensor, b: torch.Tensor) -> float:
    """``||a - b|| / max(||a||, ||b||)``, with the scale of the reference."""
    a = a.detach().reshape(-1).double()
    b = b.detach().reshape(-1).double()
    denominator = max(float(a.norm()), float(b.norm()), 1e-300)
    return float((a - b).norm()) / denominator


def directional_derivative_ad(f, x: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """Autograd directional derivative ``d/dt f(x + t v) at t = 0``."""
    x_d = x.detach().clone().requires_grad_(True)
    value = f(x_d)
    if value.numel() != 1:
        raise ValueError("f must return a scalar")
    (grad,) = torch.autograd.grad(value, x_d)
    return (grad * direction).sum()


def central_difference_directional(f, x: torch.Tensor, direction: torch.Tensor,
                                   eps: float = 1e-6) -> float:
    """Central finite difference of ``f`` along ``direction`` (detached)."""
    with torch.no_grad():
        plus = float(f((x + eps * direction).detach()))
        minus = float(f((x - eps * direction).detach()))
    return (plus - minus) / (2.0 * eps)


def central_difference_gradient(f, x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Componentwise central-difference gradient of a scalar function."""
    grad = torch.zeros_like(x, dtype=torch.float64)
    flat_x = x.detach().reshape(-1)
    flat_g = grad.reshape(-1)
    for index in range(flat_x.numel()):
        step = torch.zeros_like(flat_x)
        step[index] = eps
        step = step.reshape(x.shape)
        with torch.no_grad():
            plus = float(f((x + step).detach()))
            minus = float(f((x - step).detach()))
        flat_g[index] = (plus - minus) / (2.0 * eps)
    return grad


def gradient_consistency(f, x: torch.Tensor, n_directions: int = 8,
                         eps: float = 1e-6, seed: int = 0) -> dict:
    """Compare autograd and central-difference directional derivatives.

    Returns a dictionary with, for every random direction, the autograd value,
    the finite-difference value and the relative difference, plus the summary
    statistics used by the verification benchmark and the regression tests.
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    rows = []
    for _ in range(n_directions):
        direction = torch.randn(x.shape, generator=generator, dtype=torch.float64).to(x.device)
        direction = direction / direction.norm()
        ad = float(directional_derivative_ad(f, x, direction))
        fd = central_difference_directional(f, x, direction, eps=eps)
        scale = max(abs(ad), abs(fd), 1e-300)
        rows.append({
            "autograd": ad,
            "finite_difference": fd,
            "relative_difference": abs(ad - fd) / scale,
        })
    worst = max(row["relative_difference"] for row in rows)
    return {
        "directions": rows,
        "eps": eps,
        "max_relative_difference": worst,
        "mean_relative_difference": sum(r["relative_difference"] for r in rows) / len(rows),
        "pass": worst < 1e-6,
    }# ---------------------------------------------------------------------------
# Diffusive (SIPG + Crank--Nicolson) integration
# ---------------------------------------------------------------------------

def heat_1d_operators(ncells: int, degree: int, penalty: float = 20.0,
                      device=None, dtype=torch.float64):
    """Assemble the SIPG mass and unit-diffusivity matrices used by ``heat_1d``.

    The assembly mirrors :func:`tensordg.heat_1d.solve_heat_1d` block by block
    (volume terms, interior faces, Dirichlet boundary faces) so that a frozen
    step sequence reproduces that solver bit for bit.  The diffusivity ``alpha``
    is *not* folded in here: keeping the operator unit-scaled lets ``alpha`` be a
    differentiable scalar inside :func:`advance_heat_1d`.

    Returns ``(mass_matrix, diffusion, centers)`` with ``mass_matrix`` such that
    ``u^T mass_matrix u == int_0^1 u_h^2 dx`` for the modal vector ``u``.
    """
    from .heat_1d import _operators as _heat_operators

    if ncells < 2 or degree < 1 or penalty <= 0.0:
        raise ValueError("ncells >= 2, degree >= 1, and positive penalty are required")
    if device is None:
        device = torch.device("cpu")
    dev = torch.device(device)
    h = 1.0 / ncells
    centers = (torch.arange(ncells, device=dev, dtype=dtype) + 0.5) * h
    (
        _nodes, _weights, phi, mass, stiffness, phi_left, phi_right,
        dphi_left, dphi_right,
    ) = _heat_operators(degree, dev, dtype)
    modes = degree + 1
    ndofs = ncells * modes
    diffusion = torch.zeros((ndofs, ndofs), device=dev, dtype=dtype)
    mass_matrix = torch.zeros_like(diffusion)
    local = torch.arange(modes, device=dev)

    def dofs(cell):
        return cell * modes + local

    def add_block(matrix, indices, block):
        matrix[indices[:, None], indices[None, :]] += block

    volume_diffusion = (2.0 / h) * stiffness
    volume_mass = 0.5 * h * mass
    for cell in range(ncells):
        indices = dofs(cell)
        add_block(diffusion, indices, volume_diffusion)
        add_block(mass_matrix, indices, volume_mass)

    jump = torch.cat((phi_right, -phi_left))
    average_derivative = 0.5 * (2.0 / h) * torch.cat((dphi_right, dphi_left))
    consistency = -torch.outer(jump, average_derivative)
    face = (
        consistency
        + consistency.T
        + penalty * modes * modes / h * torch.outer(jump, jump)
    )
    for cell in range(ncells - 1):
        indices = torch.cat((dofs(cell), dofs(cell + 1)))
        add_block(diffusion, indices, face)

    for cell, endpoint, derivative_endpoint, outward in (
        (0, phi_left, dphi_left, -1.0),
        (ncells - 1, phi_right, dphi_right, 1.0),
    ):
        derivative = outward * (2.0 / h) * derivative_endpoint
        consistency = -torch.outer(endpoint, derivative)
        block = (
            consistency
            + consistency.T
            + penalty * modes * modes / h * torch.outer(endpoint, endpoint)
        )
        add_block(diffusion, dofs(cell), block)
    return mass_matrix, diffusion, centers


def advance_heat_1d(coeff0: torch.Tensor, ncells: int, degree: int,
                    nsteps: int = 1, dt=0.0, alpha=1.0,
                    penalty: float = 20.0) -> torch.Tensor:
    """Crank--Nicolson integration of the SIPG heat operator ``u_t = alpha u_xx``.

    ``coeff0`` is ``[ncells, degree+1]`` (modal Legendre coefficients, zero
    Dirichlet data at both ends, matching :func:`tensordg.heat_1d.solve_heat_1d`).
    ``dt`` is *frozen* (scalar or per-step sequence), so the returned derivative
    is the tangent of the discrete scheme at that fixed step size -- exactly what
    the finite-difference reference evaluates.  ``alpha`` may be a 0-dim tensor,
    which makes ``d(objective)/d(alpha)`` available through the linear solves.

    The diffusive half of the TensorDG operator zoo was previously outside the
    differentiable entry points (``advance_advection_1d`` / ``advance_euler_*``
    cover the hyperbolic solvers only).
    """
    if coeff0.dim() != 2 or coeff0.shape[0] != ncells or coeff0.shape[1] != degree + 1:
        raise ValueError("coeff0 must have shape [ncells, degree+1]")
    mass_matrix, diffusion, _centers = heat_1d_operators(
        ncells, degree, penalty, coeff0.device, coeff0.dtype
    )
    if torch.is_tensor(alpha):
        alpha_t = alpha.to(dtype=coeff0.dtype)
    else:
        alpha_t = torch.as_tensor(float(alpha), device=coeff0.device, dtype=coeff0.dtype)
    state = coeff0.reshape(-1)
    for step in _step_sizes(dt, nsteps):
        if step < 0.0:
            raise ValueError("dt must be non-negative")
        system = mass_matrix + (0.5 * step * alpha_t) * diffusion
        explicit = mass_matrix - (0.5 * step * alpha_t) * diffusion
        state = torch.linalg.solve(system, explicit @ state)
    return state.reshape(ncells, degree + 1)