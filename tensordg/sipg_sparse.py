"""Sparse SIPG assembly and solve for elliptic problems.

Replaces the dense matrix in ``poisson_2d`` / ``heat_1d`` with a scipy CSR
matrix so that problems with >10^4 DOFs become tractable.  The assembly
logic mirrors the dense path exactly (same local operators, same face
integrals) so that results match to machine precision.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch


# ---------------------------------------------------------------------------
# 1-D assembly helpers (shared by heat_1d sparse / matrix-free paths)
# ---------------------------------------------------------------------------

def assemble_heat_1d_sparse(
    ncells: int,
    degree: int,
    penalty: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple:
    """Assemble the SIPG diffusion + mass matrices for 1-D heat as scipy CSR.

    Returns ``(diffusion_csr, mass_csr, operators)`` where *operators* is the
    tuple ``(nodes, weights, phi, mass_dense, stiffness_dense, ...)`` needed
    by downstream code (RHS projection, error evaluation).
    """
    from tensordg.heat_1d import _operators

    h = 1.0 / ncells
    (
        nodes, weights, phi, mass, stiffness, phi_left, phi_right,
        dphi_left, dphi_right,
    ) = _operators(degree, device, dtype)
    modes = degree + 1
    ndofs = ncells * modes

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals_diff: list[np.ndarray] = []
    vals_mass: list[np.ndarray] = []

    local = torch.arange(modes, device=device)

    def _dofs(cell):
        return cell * modes + local

    volume_diffusion = (2.0 / h) * stiffness
    volume_mass = 0.5 * h * mass

    for cell in range(ncells):
        idx = _dofs(cell)
        ii, jj = torch.meshgrid(idx, idx, indexing="ij")
        rows.append(ii.reshape(-1).cpu().numpy())
        cols.append(jj.reshape(-1).cpu().numpy())
        vals_diff.append(volume_diffusion.reshape(-1).cpu().numpy())
        vals_mass.append(volume_mass.reshape(-1).cpu().numpy())

    # Interior faces
    jump = torch.cat((phi_right, -phi_left))
    average_derivative = 0.5 * (2.0 / h) * torch.cat((dphi_right, dphi_left))
    consistency = -torch.outer(jump, average_derivative)
    face = consistency + consistency.T + penalty * modes * modes / h * torch.outer(jump, jump)

    for cell in range(ncells - 1):
        idx = torch.cat((_dofs(cell), _dofs(cell + 1)))
        ii, jj = torch.meshgrid(idx, idx, indexing="ij")
        rows.append(ii.reshape(-1).cpu().numpy())
        cols.append(jj.reshape(-1).cpu().numpy())
        vals_diff.append(face.reshape(-1).cpu().numpy())
        vals_mass.append(np.zeros(idx.numel() ** 2))

    # Boundary faces
    for cell, endpoint, deriv_endpoint, outward in (
        (0, phi_left, dphi_left, -1.0),
        (ncells - 1, phi_right, dphi_right, 1.0),
    ):
        derivative = outward * (2.0 / h) * deriv_endpoint
        consistency_b = -torch.outer(endpoint, derivative)
        block = (
            consistency_b + consistency_b.T
            + penalty * modes * modes / h * torch.outer(endpoint, endpoint)
        )
        idx = _dofs(cell)
        ii, jj = torch.meshgrid(idx, idx, indexing="ij")
        rows.append(ii.reshape(-1).cpu().numpy())
        cols.append(jj.reshape(-1).cpu().numpy())
        vals_diff.append(block.reshape(-1).cpu().numpy())
        vals_mass.append(np.zeros(idx.numel() ** 2))

    all_rows = np.concatenate(rows)
    all_cols = np.concatenate(cols)
    all_diff = np.concatenate(vals_diff)
    all_mass = np.concatenate(vals_mass)

    diffusion_csr = sp.csr_matrix((all_diff, (all_rows, all_cols)), shape=(ndofs, ndofs))
    mass_csr = sp.csr_matrix((all_mass, (all_rows, all_cols)), shape=(ndofs, ndofs))

    operators = (
        nodes, weights, phi, mass, stiffness, phi_left, phi_right,
        dphi_left, dphi_right,
    )
    return diffusion_csr, mass_csr, operators


# ---------------------------------------------------------------------------
# 2-D assembly helper
# ---------------------------------------------------------------------------

def assemble_poisson_2d_sparse(
    nx: int,
    ny: int,
    degree: int,
    penalty: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple:
    """Assemble the 2-D SIPG stiffness matrix as scipy CSR.

    Returns ``(stiffness_csr, operators)``.
    """
    from tensordg.poisson_2d import _face_basis, _operators

    hx, hy = 1.0 / nx, 1.0 / ny
    (
        nodes, weights, phi, mass, stiffness, phi_left, phi_right,
        dphi_left, dphi_right,
    ) = _operators(degree, device, dtype)
    modes = degree + 1
    local_dofs = modes * modes
    ndofs = nx * ny * local_dofs

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []

    local = torch.arange(local_dofs, device=device)

    def _dofs(ix, iy):
        return (ix * ny + iy) * local_dofs + local

    # Volume
    volume_x = (hy / hx) * torch.einsum("ki,lj->klij", stiffness, mass).reshape(local_dofs, local_dofs)
    volume_y = (hx / hy) * torch.einsum("ki,lj->klij", mass, stiffness).reshape(local_dofs, local_dofs)
    volume = volume_x + volume_y

    for ix in range(nx):
        for iy in range(ny):
            idx = _dofs(ix, iy)
            ii, jj = torch.meshgrid(idx, idx, indexing="ij")
            rows.append(ii.reshape(-1).cpu().numpy())
            cols.append(jj.reshape(-1).cpu().numpy())
            vals.append(volume.reshape(-1).cpu().numpy())

    # Interior vertical faces
    trace_right, deriv_right_f = _face_basis(phi, phi_right, dphi_right, "x", 2.0 / hx)
    trace_left, deriv_left_f = _face_basis(phi, phi_left, dphi_left, "x", 2.0 / hx)
    jump_x = torch.cat((trace_right, -trace_left), dim=1)
    avg_deriv_x = 0.5 * torch.cat((deriv_right_f, deriv_left_f), dim=1)
    measure_x = 0.5 * hy
    cons_x = -measure_x * torch.einsum("q,qk,ql->kl", weights, jump_x, avg_deriv_x)
    pen_x = penalty * modes * modes / hx
    face_x = cons_x + cons_x.T + pen_x * measure_x * torch.einsum("q,qk,ql->kl", weights, jump_x, jump_x)

    for ix in range(nx - 1):
        for iy in range(ny):
            idx = torch.cat((_dofs(ix, iy), _dofs(ix + 1, iy)))
            ii, jj = torch.meshgrid(idx, idx, indexing="ij")
            rows.append(ii.reshape(-1).cpu().numpy())
            cols.append(jj.reshape(-1).cpu().numpy())
            vals.append(face_x.reshape(-1).cpu().numpy())

    # Interior horizontal faces
    trace_top, deriv_top_f = _face_basis(phi, phi_right, dphi_right, "y", 2.0 / hy)
    trace_bot, deriv_bot_f = _face_basis(phi, phi_left, dphi_left, "y", 2.0 / hy)
    jump_y = torch.cat((trace_top, -trace_bot), dim=1)
    avg_deriv_y = 0.5 * torch.cat((deriv_top_f, deriv_bot_f), dim=1)
    measure_y = 0.5 * hx
    cons_y = -measure_y * torch.einsum("q,qk,ql->kl", weights, jump_y, avg_deriv_y)
    pen_y = penalty * modes * modes / hy
    face_y = cons_y + cons_y.T + pen_y * measure_y * torch.einsum("q,qk,ql->kl", weights, jump_y, jump_y)

    for ix in range(nx):
        for iy in range(ny - 1):
            idx = torch.cat((_dofs(ix, iy), _dofs(ix, iy + 1)))
            ii, jj = torch.meshgrid(idx, idx, indexing="ij")
            rows.append(ii.reshape(-1).cpu().numpy())
            cols.append(jj.reshape(-1).cpu().numpy())
            vals.append(face_y.reshape(-1).cpu().numpy())

    # Boundary faces (x)
    for ix, endpoint, deriv_endpoint, outward in (
        (0, phi_left, dphi_left, -1.0),
        (nx - 1, phi_right, dphi_right, 1.0),
    ):
        trace, derivative = _face_basis(phi, endpoint, deriv_endpoint, "x", 2.0 / hx, outward)
        cons_b = -measure_x * torch.einsum("q,qk,ql->kl", weights, trace, derivative)
        block = cons_b + cons_b.T + pen_x * measure_x * torch.einsum("q,qk,ql->kl", weights, trace, trace)
        for iy in range(ny):
            idx = _dofs(ix, iy)
            ii, jj = torch.meshgrid(idx, idx, indexing="ij")
            rows.append(ii.reshape(-1).cpu().numpy())
            cols.append(jj.reshape(-1).cpu().numpy())
            vals.append(block.reshape(-1).cpu().numpy())

    # Boundary faces (y)
    for iy, endpoint, deriv_endpoint, outward in (
        (0, phi_left, dphi_left, -1.0),
        (ny - 1, phi_right, dphi_right, 1.0),
    ):
        trace, derivative = _face_basis(phi, endpoint, deriv_endpoint, "y", 2.0 / hy, outward)
        cons_b = -measure_y * torch.einsum("q,qk,ql->kl", weights, trace, derivative)
        block = cons_b + cons_b.T + pen_y * measure_y * torch.einsum("q,qk,ql->kl", weights, trace, trace)
        for ix in range(nx):
            idx = _dofs(ix, iy)
            ii, jj = torch.meshgrid(idx, idx, indexing="ij")
            rows.append(ii.reshape(-1).cpu().numpy())
            cols.append(jj.reshape(-1).cpu().numpy())
            vals.append(block.reshape(-1).cpu().numpy())

    all_rows = np.concatenate(rows)
    all_cols = np.concatenate(cols)
    all_vals = np.concatenate(vals)

    stiffness_csr = sp.csr_matrix((all_vals, (all_rows, all_cols)), shape=(ndofs, ndofs))

    operators = (
        nodes, weights, phi, mass, stiffness, phi_left, phi_right,
        dphi_left, dphi_right,
    )
    return stiffness_csr, operators


# ---------------------------------------------------------------------------
# Sparse solve wrappers
# ---------------------------------------------------------------------------

def solve_poisson_2d_sparse(
    ncells: int | tuple[int, int] = 16,
    degree: int = 1,
    penalty: float = 10.0,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> tuple:
    """Sparse SIPG Poisson solve.  Signature mirrors ``solve_poisson_2d``."""
    from tensordg.poisson_2d import _operators

    nx, ny = (ncells, ncells) if isinstance(ncells, int) else ncells
    dev = torch.device(device)
    hx, hy = 1.0 / nx, 1.0 / ny
    x = (torch.arange(nx, device=dev, dtype=dtype) + 0.5) * hx
    y = (torch.arange(ny, device=dev, dtype=dtype) + 0.5) * hy

    stiffness_csr, operators = assemble_poisson_2d_sparse(
        nx, ny, degree, penalty, dev, dtype,
    )
    nodes, weights, phi, mass, stiffness, *_ = operators
    modes = degree + 1

    # RHS
    xq = x[:, None, None, None] + 0.5 * hx * nodes[None, None, :, None]
    yq = y[None, :, None, None] + 0.5 * hy * nodes[None, None, None, :]
    forcing = 2.0 * math.pi ** 2 * torch.sin(math.pi * xq) * torch.sin(math.pi * yq)
    load = 0.25 * hx * hy * torch.einsum(
        "xyqr,q,r,qk,rl->xykl", forcing, weights, weights, phi, phi,
    )
    rhs_np = load.reshape(-1).cpu().numpy().astype(np.float64)

    solution_np = spla.spsolve(stiffness_csr, rhs_np)
    solution = torch.as_tensor(solution_np, device=dev, dtype=dtype)
    coeff = solution.reshape(nx, ny, modes, modes)

    symmetry_error = float(np.max(np.abs(stiffness_csr - stiffness_csr.T)))
    metadata = {
        "ncells": (nx, ny), "degree": degree,
        "dofs": stiffness_csr.shape[0],
        "nnz": stiffness_csr.nnz,
        "penalty": penalty, "matrix_symmetry_error": symmetry_error,
        "device": str(dev), "backend": "sparse",
    }
    return x, y, coeff, metadata


def solve_heat_1d_sparse(
    ncells: int = 16,
    degree: int = 2,
    final_time: float = 0.1,
    time_steps: int | None = None,
    penalty: float = 20.0,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> tuple:
    """Sparse SIPG heat solve.  Signature mirrors ``solve_heat_1d``."""
    from tensordg.heat_1d import _operators, exact_temperature

    dev = torch.device(device)
    h = 1.0 / ncells
    centers = (torch.arange(ncells, device=dev, dtype=dtype) + 0.5) * h

    diffusion_csr, mass_csr, operators = assemble_heat_1d_sparse(
        ncells, degree, penalty, dev, dtype,
    )
    nodes, weights, phi, mass, stiffness, *_ = operators
    modes = degree + 1

    # Initial condition
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

    state = coeff.reshape(-1).cpu().numpy().astype(np.float64)

    if steps:
        system = mass_csr + 0.5 * dt * diffusion_csr
        explicit = mass_csr - 0.5 * dt * diffusion_csr
        # Factorise once
        lu = spla.splu(system.tocsc())
        for _ in range(steps):
            state = lu.solve(explicit @ state)

    solution = torch.as_tensor(state, device=dev, dtype=dtype)
    coeff = solution.reshape(ncells, modes)

    symmetry_error = float(np.max(np.abs(diffusion_csr - diffusion_csr.T)))
    metadata = {
        "ncells": ncells, "degree": degree, "final_time": final_time,
        "steps": steps, "dt": dt, "penalty": penalty,
        "dofs": diffusion_csr.shape[0], "nnz": diffusion_csr.nnz,
        "diffusion_symmetry_error": symmetry_error,
        "device": str(dev), "backend": "sparse",
    }
    return centers, coeff, metadata
