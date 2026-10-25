"""One-dimensional heat equation verified with SIPG and Crank--Nicolson time stepping.

Three backends are supported via the ``backend`` parameter:
  * ``"dense"``        -- assemble dense torch matrices, solve with torch.linalg.solve
  * ``"sparse"``       -- assemble scipy CSR matrices, solve with scipy.sparse.linalg.splu
  * ``"matrix-free"``  -- no global assembly, solve with preconditioned CG
"""
from __future__ import annotations

import math

import numpy as np
import torch


def exact_temperature(x: torch.Tensor, final_time) -> torch.Tensor:
    time = torch.as_tensor(final_time, device=x.device, dtype=x.dtype)
    return torch.sin(math.pi * x) * torch.exp(-math.pi ** 2 * time)


def _operators(degree: int, device: torch.device, dtype: torch.dtype):
    nq = max(2 * degree + 4, 6)
    nodes_np, weights_np = np.polynomial.legendre.leggauss(nq)
    nodes = torch.as_tensor(nodes_np, device=device, dtype=dtype)
    weights = torch.as_tensor(weights_np, device=device, dtype=dtype)
    values, derivatives = [], []
    left, right, deriv_left, deriv_right = [], [], [], []
    for k in range(degree + 1):
        polynomial = np.zeros(k + 1)
        polynomial[-1] = 1.0
        derivative = np.polynomial.legendre.legder(polynomial)
        values.append(torch.as_tensor(
            np.polynomial.legendre.legval(nodes_np, polynomial), device=device, dtype=dtype,
        ))
        deriv_values = (
            np.polynomial.legendre.legval(nodes_np, derivative)
            if len(derivative) else np.zeros_like(nodes_np)
        )
        derivatives.append(torch.as_tensor(deriv_values, device=device, dtype=dtype))
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
    mass = torch.einsum("q,qk,qi->ki", weights, phi, phi)
    stiffness = torch.einsum("q,qk,qi->ki", weights, dphi, dphi)
    tensor = lambda data: torch.as_tensor(data, device=device, dtype=dtype)
    return (
        nodes, weights, phi, mass, stiffness, tensor(left), tensor(right),
        tensor(deriv_left), tensor(deriv_right),
    )


def _add_block(matrix, indices, block):
    matrix[indices[:, None], indices[None, :]] += block


def solve_heat_1d(
    ncells: int = 16,
    degree: int = 2,
    final_time: float = 0.1,
    time_steps: int | None = None,
    penalty: float = 20.0,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    backend: str = "dense",
):
    """Solve u_t=u_xx with zero Dirichlet data and sin(pi*x) initial data.

    Crank--Nicolson is used for the global method-of-lines system. If
    ``time_steps`` is omitted, dt is approximately ``0.02*cell_width``.

    Parameters
    ----------
    backend : str
        - ``"dense"``: assemble dense torch matrices (default, backward compatible)
        - ``"sparse"``: assemble scipy CSR matrices, solve with splu
        - ``"matrix-free"``: no global assembly, solve with PCG per step
    """
    if backend not in ("dense", "sparse", "matrix-free"):
        raise ValueError("backend must be 'dense', 'sparse', or 'matrix-free'")

    if backend == "sparse":
        from tensordg.sipg_sparse import solve_heat_1d_sparse
        return solve_heat_1d_sparse(
            ncells, degree, final_time, time_steps, penalty, device, dtype,
        )
    elif backend == "matrix-free":
        from tensordg.sipg_matrixfree import solve_heat_1d_matrixfree
        return solve_heat_1d_matrixfree(
            ncells, degree, final_time, time_steps, penalty, device, dtype,
        )

    # backend == "dense": original implementation
    if ncells < 2 or degree not in (1, 2, 3) or penalty <= 0.0:
        raise ValueError("ncells >= 2, degree in [1,3], and positive penalty are required")
    if final_time < 0.0 or (time_steps is not None and time_steps < 1):
        raise ValueError("final_time must be >= 0 and time_steps must be positive")
    dev = torch.device(device)
    h = 1.0 / ncells
    centers = (torch.arange(ncells, device=dev, dtype=dtype) + 0.5) * h
    (
        nodes, weights, phi, mass, stiffness, phi_left, phi_right,
        dphi_left, dphi_right,
    ) = _operators(degree, dev, dtype)
    modes = degree + 1
    ndofs = ncells * modes
    diffusion = torch.zeros((ndofs, ndofs), device=dev, dtype=dtype)
    mass_matrix = torch.zeros_like(diffusion)
    local = torch.arange(modes, device=dev)

    def dofs(cell):
        return cell * modes + local

    volume_diffusion = (2.0 / h) * stiffness
    volume_mass = 0.5 * h * mass
    for cell in range(ncells):
        indices = dofs(cell)
        _add_block(diffusion, indices, volume_diffusion)
        _add_block(mass_matrix, indices, volume_mass)

    jump = torch.cat((phi_right, -phi_left))
    average_derivative = 0.5 * (2.0 / h) * torch.cat((dphi_right, dphi_left))
    consistency = -torch.outer(jump, average_derivative)
    face = consistency + consistency.T + penalty * modes * modes / h * torch.outer(jump, jump)
    for cell in range(ncells - 1):
        indices = torch.cat((dofs(cell), dofs(cell + 1)))
        _add_block(diffusion, indices, face)

    for cell, endpoint, derivative_endpoint, outward in (
        (0, phi_left, dphi_left, -1.0),
        (ncells - 1, phi_right, dphi_right, 1.0),
    ):
        derivative = outward * (2.0 / h) * derivative_endpoint
        consistency = -torch.outer(endpoint, derivative)
        block = consistency + consistency.T + penalty * modes * modes / h * torch.outer(
            endpoint, endpoint,
        )
        _add_block(diffusion, dofs(cell), block)

    xq = centers[:, None] + 0.5 * h * nodes[None, :]
    initial = torch.sin(math.pi * xq)
    coeff = torch.empty((ncells, modes), device=dev, dtype=dtype)
    for k in range(modes):
        coeff[:, k] = (2 * k + 1) * 0.5 * torch.sum(
            weights[None, :] * initial * phi[:, k][None, :], dim=1,
        )
    steps = (
        max(1, math.ceil(final_time / (0.02 * h)))
        if time_steps is None else time_steps
    ) if final_time > 0.0 else 0
    dt = final_time / steps if steps else 0.0
    state = coeff.reshape(-1)
    if steps:
        system = mass_matrix + 0.5 * dt * diffusion
        explicit = mass_matrix - 0.5 * dt * diffusion
        for _ in range(steps):
            state = torch.linalg.solve(system, explicit @ state)
    coeff = state.reshape(ncells, modes)
    metadata = {
        "ncells": ncells, "degree": degree, "final_time": final_time,
        "steps": steps, "dt": dt, "penalty": penalty, "dofs": ndofs,
        "diffusion_symmetry_error": float(torch.max(torch.abs(diffusion - diffusion.T)).item()),
        "device": str(dev),
    }
    return centers, coeff, metadata


def heat_error(
    centers: torch.Tensor, coeff: torch.Tensor, final_time: float,
) -> dict[str, float]:
    degree = coeff.shape[1] - 1
    h = 1.0 / centers.numel()
    nodes_np, weights_np = np.polynomial.legendre.leggauss(max(2 * degree + 5, 8))
    nodes = torch.as_tensor(nodes_np, device=coeff.device, dtype=coeff.dtype)
    weights = torch.as_tensor(weights_np, device=coeff.device, dtype=coeff.dtype)
    basis = []
    for k in range(degree + 1):
        polynomial = np.zeros(k + 1)
        polynomial[-1] = 1.0
        basis.append(torch.as_tensor(
            np.polynomial.legendre.legval(nodes_np, polynomial),
            device=coeff.device, dtype=coeff.dtype,
        ))
    phi = torch.stack(basis, dim=1)
    numerical = coeff @ phi.T
    x = centers[:, None] + 0.5 * h * nodes[None, :]
    exact = exact_temperature(x, final_time)
    error2 = 0.5 * h * torch.sum(weights[None, :] * (numerical - exact) ** 2)
    norm2 = 0.5 * h * torch.sum(weights[None, :] * exact ** 2)
    absolute = float(torch.sqrt(error2).item())
    norm = float(torch.sqrt(norm2).item())
    return {"absolute_l2": absolute, "exact_l2": norm, "relative_l2": absolute / norm}
