"""Steady one-dimensional advection--diffusion DG benchmark with an exact layer solution.

The merged TensorDG operator stack has two separately verified halves -- upwind DG
advection (:mod:`tensordg.advection_1d`) and symmetric interior penalty (SIPG)
diffusion (:mod:`tensordg.heat_1d`, :mod:`tensordg.sipg_sparse`,
:mod:`tensordg.sipg_matrixfree`) -- but the two were never exercised *together*,
and the 1-D SIPG solvers only ever ran with **homogeneous** Dirichlet data.  This
module closes that gap with the classical exponential-boundary-layer problem

    a u'(x) - D u''(x) = 0,   u(0) = g_L,  u(1) = g_R,   a > 0, D > 0,      (1)

whose exact solution is

    u(x) = g_L + (g_R - g_L) * (exp(Pe x) - 1) / (exp(Pe) - 1),
    Pe  = a L / D                                                          (2)

and whose exact *total flux* is constant in x,

    J = a u - D u' = a * (g_R - g_L) / (1 - exp(-Pe)).                      (3)

Equation (3) is the discrete-consistency meter used by the tests: for a
consistent DG discretisation the fluxes recovered on every cell face agree with
each other (and with (3)) to discretisation accuracy, so it detects broken face
coupling that an L2 error norm can hide.

Discretisation
--------------
Modal Legendre basis (identical conventions to :mod:`tensordg.advection_1d` and
:mod:`tensordg.heat_1d`), upwind flux for the advective face term, SIPG for the
diffusive term with the penalty ``penalty * (degree+1)**2 / h`` used by
``heat_1d``, and inhomogeneous Dirichlet data imposed through the Nitsche-type
boundary terms.  Everything is assembled as one dense global system, so the
solver stays O(N^3) with N = ncells*(degree+1) and is meant for verification on
small grids, not for production runs.
"""
from __future__ import annotations

import math

import numpy as np
import torch


def exact_solution(x: torch.Tensor, peclet: float) -> torch.Tensor:
    """Exact solution (2) of (1) with ``g_L = 0``, ``g_R = 1``."""
    decay = math.exp(-peclet)
    return (torch.exp(peclet * (x - 1.0)) - decay) / (1.0 - decay)


def exact_flux(
    peclet: float, speed: float = 1.0, left_value: float = 0.0, right_value: float = 1.0,
) -> float:
    """Exact constant total flux (3): ``J = a*[g_L - (g_R - g_L)/(exp(Pe) - 1)]``."""
    decay = math.exp(-peclet)
    jump = (right_value - left_value) * decay / (1.0 - decay)
    return speed * (left_value - jump)


def _operators(degree: int, device: torch.device, dtype: torch.dtype):
    nq = max(2 * degree + 4, 6)
    nodes_np, weights_np = np.polynomial.legendre.leggauss(nq)
    nodes = torch.as_tensor(nodes_np, device=device, dtype=dtype)
    weights = torch.as_tensor(weights_np, device=device, dtype=dtype)
    values, derivatives, left, right, deriv_left, deriv_right = [], [], [], [], [], []
    for k in range(degree + 1):
        polynomial = np.zeros(k + 1)
        polynomial[-1] = 1.0
        derivative = np.polynomial.legendre.legder(polynomial)
        values.append(torch.as_tensor(
            np.polynomial.legendre.legval(nodes_np, polynomial), device=device, dtype=dtype,
        ))
        derivatives.append(torch.as_tensor(
            np.polynomial.legendre.legval(nodes_np, derivative)
            if len(derivative) else np.zeros_like(nodes_np),
            device=device, dtype=dtype,
        ))
        left.append((-1.0) ** k)
        right.append(1.0)
        deriv_left.append(
            float(np.polynomial.legendre.legval(-1.0, derivative)) if len(derivative) else 0.0,
        )
        deriv_right.append(
            float(np.polynomial.legendre.legval(1.0, derivative)) if len(derivative) else 0.0,
        )
    phi = torch.stack(values, dim=1)
    dphi = torch.stack(derivatives, dim=1)
    stiffness = torch.einsum("q,qk,qi->ki", weights, dphi, dphi)
    # volume[i, j] = int_{-1}^{1} P_j dP_i/dxi dxi  (weak advective volume term)
    volume = torch.einsum("q,qj,qi->ij", weights, phi, dphi)
    tensor = lambda data: torch.as_tensor(data, device=device, dtype=dtype)
    return (
        nodes, weights, phi, stiffness, volume, tensor(left), tensor(right),
        tensor(deriv_left), tensor(deriv_right),
    )


def solve_advection_diffusion_1d(
    ncells: int = 32,
    degree: int = 2,
    peclet: float = 10.0,
    speed: float = 1.0,
    penalty: float = 20.0,
    left_value: float = 0.0,
    right_value: float = 1.0,
    advection_scheme: str = "upwind",
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
):
    """Solve (1) on [0, 1] and return (cell_centres, modal_coefficients, metadata).

    Parameters
    ----------
    peclet : float
        Peclet number ``a L / D`` with ``L = 1``; the diffusivity is
        ``D = speed / peclet``.
    advection_scheme : {"upwind", "downwind"}
        ``"downwind"`` exists only as an implementation negative control: it is
        *not* a consistent scheme and must fail the verification gates.
    """
    if ncells < 2 or degree not in (1, 2, 3):
        raise ValueError("ncells >= 2 and degree in [1, 2, 3] are required")
    if peclet <= 0.0 or speed == 0.0:
        raise ValueError("peclet > 0 and speed != 0 are required")
    if penalty <= 0.0:
        raise ValueError("penalty must be positive")
    if advection_scheme not in ("upwind", "downwind"):
        raise ValueError("advection_scheme must be 'upwind' or 'downwind'")

    dev = torch.device(device)
    h = 1.0 / ncells
    modes = degree + 1
    ndofs = ncells * modes
    centers = (torch.arange(ncells, device=dev, dtype=dtype) + 0.5) * h
    (
        nodes, weights, phi, stiffness, volume, phi_left, phi_right,
        dphi_left, dphi_right,
    ) = _operators(degree, dev, dtype)
    diffusivity = speed / peclet
    a = float(speed)

    system = torch.zeros((ndofs, ndofs), device=dev, dtype=dtype)
    rhs = torch.zeros(ndofs, device=dev, dtype=dtype)
    local = torch.arange(modes, device=dev)
    dofs = lambda cell: cell * modes + local
    def add(rows, cols, block):
        system[rows[:, None], cols[None, :]] += block

    # ---- volume terms: SIPG diffusion + weak advection ----------------------
    # Steady residual convention:  B_diff(u, v) - A_adv(u, v) = L_diff(v), where
    # B_diff is the SIPG form of -D u'' (same matrix as tensordg.heat_1d) and
    # A_adv(u, v_i) = a * int_K u v_i' - a * f_hat_R v_i(1) + a * f_hat_L v_i(-1).
    # The two blocks therefore carry *opposite* signs.
    diffusion_volume = diffusivity * (2.0 / h) * stiffness
    advection_volume = -a * volume
    for cell in range(ncells):
        idx = dofs(cell)
        add(idx, idx, diffusion_volume + advection_volume)

    # ---- interior diffusive faces: SIPG face block (as in tensordg.heat_1d) --
    jump = torch.cat((phi_right, -phi_left))
    average_derivative = 0.5 * diffusivity * (2.0 / h) * torch.cat((dphi_right, dphi_left))
    consistency = -torch.outer(jump, average_derivative)
    penalty_face = penalty * modes * modes / h
    diffusion_face = consistency + consistency.T + penalty_face * torch.outer(jump, jump)
    for cell in range(ncells - 1):
        idx = torch.cat((dofs(cell), dofs(cell + 1)))
        add(idx, idx, diffusion_face)

    # ---- advective faces: one single-valued flux per face -------------------
    # The flux of face f is a * trace(source).  For the upwind scheme the source
    # is the element on the inflow side; the outflow boundary therefore uses the
    # interior trace while the inflow boundary uses the prescribed data.
    upwind = advection_scheme == "upwind"
    for face in range(ncells + 1):
        left_cell = face - 1 if face > 0 else None
        right_cell = face if face < ncells else None
        src_is_left = (a > 0.0) == upwind
        source = left_cell if src_is_left else right_cell
        trace = phi_right if src_is_left else phi_left
        if source is None:
            boundary_value = left_value if face == 0 else right_value
            if left_cell is not None:
                rhs[dofs(left_cell)] += -a * boundary_value * phi_right
            if right_cell is not None:
                rhs[dofs(right_cell)] += a * boundary_value * phi_left
            continue
        if left_cell is not None:
            add(dofs(left_cell), dofs(source), a * torch.outer(phi_right, trace))
        if right_cell is not None:
            add(dofs(right_cell), dofs(source), -a * torch.outer(phi_left, trace))

    # ---- Dirichlet boundary terms for the diffusive operator (Nitsche) ------
    for cell, endpoint, derivative_endpoint, outward, value in (
        (0, phi_left, dphi_left, -1.0, left_value),
        (ncells - 1, phi_right, dphi_right, 1.0, right_value),
    ):
        idx = dofs(cell)
        derivative = outward * diffusivity * (2.0 / h) * derivative_endpoint
        consistency = -torch.outer(endpoint, derivative)
        block = consistency + consistency.T + penalty_face * torch.outer(endpoint, endpoint)
        add(idx, idx, block)
        # data terms of the Nitsche form: -int (D v' n) g + penalty * int g v
        rhs[idx] += (-derivative + penalty_face * endpoint) * value

    state = torch.linalg.solve(system, rhs)
    coeff = state.reshape(ncells, modes)
    metadata = {
        "ncells": ncells, "degree": degree, "peclet": peclet, "speed": a,
        "diffusivity": diffusivity, "penalty": penalty, "dofs": ndofs,
        "advection_scheme": advection_scheme, "h": h,
        "device": str(dev),
    }
    return centers, coeff, metadata


def advection_diffusion_error(
    centers: torch.Tensor, coeff: torch.Tensor, peclet: float,
) -> dict[str, float]:
    """Quadrature L2 error of the discrete solution against the exact layer."""
    degree = coeff.shape[1] - 1
    h = 1.0 / centers.numel()
    nodes_np, weights_np = np.polynomial.legendre.leggauss(max(2 * degree + 5, 8))
    nodes = torch.as_tensor(nodes_np, device=coeff.device, dtype=coeff.dtype)
    weights = torch.as_tensor(weights_np, device=coeff.device, dtype=coeff.dtype)
    phi = _legendre_values(nodes_np, degree, coeff.device, coeff.dtype)
    numerical = coeff @ phi.T
    x = centers[:, None] + 0.5 * h * nodes[None, :]
    exact = exact_solution(x, peclet)
    error2 = 0.5 * h * torch.sum(weights[None, :] * (numerical - exact) ** 2)
    norm2 = 0.5 * h * torch.sum(weights[None, :] * exact ** 2)
    absolute = float(torch.sqrt(error2).item())
    norm = float(torch.sqrt(norm2).item())
    return {"absolute_l2": absolute, "exact_l2": norm, "relative_l2": absolute / norm}


def _legendre_values(nodes_np, degree: int, device, dtype) -> torch.Tensor:
    basis = []
    for k in range(degree + 1):
        polynomial = np.zeros(k + 1)
        polynomial[-1] = 1.0
        basis.append(torch.as_tensor(
            np.polynomial.legendre.legval(nodes_np, polynomial), device=device, dtype=dtype,
        ))
    return torch.stack(basis, dim=1)


def sample_solution(
    centers: torch.Tensor, coeff: torch.Tensor, samples_per_cell: int = 8,
):
    """Return ``(x, u_h)`` sampled at Gauss points inside every cell (1-D arrays)."""
    degree = coeff.shape[1] - 1
    nodes_np, _ = np.polynomial.legendre.leggauss(max(samples_per_cell, degree + 2))
    h = 1.0 / centers.numel()
    phi = _legendre_values(nodes_np, degree, coeff.device, coeff.dtype)
    x = (centers[:, None] + 0.5 * h * torch.as_tensor(
        nodes_np, device=coeff.device, dtype=coeff.dtype,
    )[None, :]).reshape(-1)
    values = (coeff @ phi.T).reshape(-1)
    return x, values


def face_fluxes(
    coeff: torch.Tensor, peclet: float, speed: float = 1.0,
) -> torch.Tensor:
    """Total flux ``a u - D u'`` evaluated at every cell's right face.

    For the exact solution every entry equals :func:`exact_flux`; the spread of
    the returned vector is therefore a direct measure of face-coupling
    consistency, independent of the L2 error norm.
    """
    degree = coeff.shape[1] - 1
    h = 1.0 / coeff.shape[0]
    right_values, right_derivatives = [], []
    for k in range(degree + 1):
        polynomial = np.zeros(k + 1)
        polynomial[-1] = 1.0
        derivative = np.polynomial.legendre.legder(polynomial)
        right_values.append(1.0)
        right_derivatives.append(
            float(np.polynomial.legendre.legval(1.0, derivative)) if len(derivative) else 0.0
        )
    p_right = torch.as_tensor(right_values, device=coeff.device, dtype=coeff.dtype)
    dp_right = torch.as_tensor(right_derivatives, device=coeff.device, dtype=coeff.dtype)
    value = coeff @ p_right
    derivative = (2.0 / h) * (coeff @ dp_right)
    diffusivity = speed / peclet
    return speed * value - diffusivity * derivative