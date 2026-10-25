"""Matrix-free SIPG elliptic solvers using preconditioned CG.

Instead of assembling a global matrix, we implement the operator A@x as a
function and solve with the conjugate-gradient method.  For Cartesian meshes
a block-Jacobi preconditioner (one block per element) provides a cheap and
effective preconditioner.

The module provides:
  * ``matvec_poisson_2d`` -- matrix-free A@x for 2-D Poisson
  * ``matvec_heat_diffusion_1d`` -- matrix-free diffusion A@x for 1-D heat
  * ``solve_poisson_2d_matrixfree`` -- CG solve wrapper
  * ``solve_heat_1d_matrixfree``    -- CN time-stepping with CG inner solve
"""
from __future__ import annotations

import math
from typing import Callable

import torch


# ---------------------------------------------------------------------------
# Preconditioned CG
# ---------------------------------------------------------------------------

def _pcg(
    matvec: Callable[[torch.Tensor], torch.Tensor],
    rhs: torch.Tensor,
    precond: Callable[[torch.Tensor], torch.Tensor] | None = None,
    maxiter: int = 10_000,
    rtol: float = 1e-12,
    atol: float = 1e-15,
) -> tuple[torch.Tensor, dict]:
    """Preconditioned conjugate gradient.  Returns ``(x, info)``."""
    if precond is None:
        precond = lambda v: v
    x = torch.zeros_like(rhs)
    r = rhs - matvec(x)
    z = precond(r)
    p = z.clone()
    rz = torch.dot(r, z)
    info = {"iterations": 0, "residual_norm": float("nan")}
    rhs_norm = torch.linalg.norm(rhs)
    if rhs_norm == 0.0:
        info["residual_norm"] = 0.0
        return x, info
    r_norm = rhs_norm
    for k in range(1, maxiter + 1):
        ap = matvec(p)
        pAp = torch.dot(p, ap)
        alpha = rz / pAp
        x = x + alpha * p
        r = r - alpha * ap
        r_norm = torch.linalg.norm(r)
        if r_norm < atol or r_norm < rtol * rhs_norm:
            info["iterations"] = k
            info["residual_norm"] = float(r_norm.item())
            return x, info
        z = precond(r)
        rz_new = torch.dot(r, z)
        beta = rz_new / rz
        p = z + beta * p
        rz = rz_new
    info["iterations"] = maxiter
    info["residual_norm"] = float(r_norm.item())
    return x, info


# ---------------------------------------------------------------------------
# Block-Jacobi preconditioner helper
# ---------------------------------------------------------------------------

def _block_jacobi_precond(
    blocks: list[torch.Tensor], block_size: int, ndofs: int,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build a block-Jacobi preconditioner from a list of dense blocks.

    Each block is inverted via Cholesky (or LU fallback) once.
    If blocks are singular, falls back to diagonal preconditioner.
    """
    inv_blocks = []
    use_diagonal = False
    for block in blocks:
        try:
            L = torch.linalg.cholesky(block)
            inv_blocks.append(torch.cholesky_inverse(L))
        except Exception:
            try:
                inv_blocks.append(torch.linalg.inv(block))
            except Exception:
                # Fall back to diagonal preconditioner
                use_diagonal = True
                break

    if use_diagonal:
        # Build diagonal preconditioner from all blocks
        diag = torch.zeros(ndofs)
        for i, block in enumerate(blocks):
            s = i * block_size
            diag[s:s + block_size] = torch.diag(block)
        # Invert non-zero entries
        diag_inv = torch.where(diag > 1e-14, 1.0 / diag, torch.zeros_like(diag))

        def apply(v: torch.Tensor) -> torch.Tensor:
            return diag_inv * v
        return apply

    def apply(v: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(v)
        for i, inv_b in enumerate(inv_blocks):
            s = i * block_size
            out[s:s + block_size] = inv_b @ v[s:s + block_size]
        return out

    return apply


# ---------------------------------------------------------------------------
# 2-D Poisson matrix-free
# ---------------------------------------------------------------------------

def matvec_poisson_2d(
    coeff_flat: torch.Tensor,
    nx: int,
    ny: int,
    degree: int,
    penalty: float,
    operators: tuple,
) -> torch.Tensor:
    """Compute A @ coeff for the 2-D SIPG Poisson problem, matrix-free."""
    from tensordg.poisson_2d import _face_basis

    (
        nodes, weights, phi, mass, stiffness, phi_left, phi_right,
        dphi_left, dphi_right,
    ) = operators
    hx, hy = 1.0 / nx, 1.0 / ny
    modes = degree + 1
    local_dofs = modes * modes
    coeff = coeff_flat.reshape(nx, ny, local_dofs)
    out = torch.zeros_like(coeff)

    # Volume contribution
    volume_x = (hy / hx) * torch.einsum(
        "ki,lj->klij", stiffness, mass,
    ).reshape(local_dofs, local_dofs)
    volume_y = (hx / hy) * torch.einsum(
        "ki,lj->klij", mass, stiffness,
    ).reshape(local_dofs, local_dofs)
    volume = volume_x + volume_y
    for ix in range(nx):
        for iy in range(ny):
            out[ix, iy] += volume @ coeff[ix, iy]

    # Interior vertical faces
    trace_right, deriv_right_f = _face_basis(
        phi, phi_right, dphi_right, "x", 2.0 / hx,
    )
    trace_left, deriv_left_f = _face_basis(
        phi, phi_left, dphi_left, "x", 2.0 / hx,
    )
    jump_x = torch.cat((trace_right, -trace_left), dim=1)
    avg_deriv_x = 0.5 * torch.cat((deriv_right_f, deriv_left_f), dim=1)
    measure_x = 0.5 * hy
    cons_x = -measure_x * torch.einsum(
        "q,qk,ql->kl", weights, jump_x, avg_deriv_x,
    )
    pen_x = penalty * modes * modes / hx
    face_x = cons_x + cons_x.T + pen_x * measure_x * torch.einsum(
        "q,qk,ql->kl", weights, jump_x, jump_x,
    )
    # Split into 4 blocks: LL, LR, RL, RR
    fLL = face_x[:local_dofs, :local_dofs]
    fLR = face_x[:local_dofs, local_dofs:]
    fRL = face_x[local_dofs:, :local_dofs]
    fRR = face_x[local_dofs:, local_dofs:]

    for ix in range(nx - 1):
        for iy in range(ny):
            uL = coeff[ix, iy]
            uR = coeff[ix + 1, iy]
            out[ix, iy] += fLL @ uL + fLR @ uR
            out[ix + 1, iy] += fRL @ uL + fRR @ uR

    # Interior horizontal faces
    trace_top, deriv_top_f = _face_basis(
        phi, phi_right, dphi_right, "y", 2.0 / hy,
    )
    trace_bot, deriv_bot_f = _face_basis(
        phi, phi_left, dphi_left, "y", 2.0 / hy,
    )
    jump_y = torch.cat((trace_top, -trace_bot), dim=1)
    avg_deriv_y = 0.5 * torch.cat((deriv_top_f, deriv_bot_f), dim=1)
    measure_y = 0.5 * hx
    cons_y = -measure_y * torch.einsum(
        "q,qk,ql->kl", weights, jump_y, avg_deriv_y,
    )
    pen_y = penalty * modes * modes / hy
    face_y = cons_y + cons_y.T + pen_y * measure_y * torch.einsum(
        "q,qk,ql->kl", weights, jump_y, jump_y,
    )
    # Split into 4 blocks: BB, BT, TB, TT
    fBB = face_y[:local_dofs, :local_dofs]
    fBT = face_y[:local_dofs, local_dofs:]
    fTB = face_y[local_dofs:, :local_dofs]
    fTT = face_y[local_dofs:, local_dofs:]

    for ix in range(nx):
        for iy in range(ny - 1):
            uB = coeff[ix, iy]
            uT = coeff[ix, iy + 1]
            out[ix, iy] += fBB @ uB + fBT @ uT
            out[ix, iy + 1] += fTB @ uB + fTT @ uT

    # Boundary faces (x)
    for ix, endpoint, deriv_endpoint, outward in (
        (0, phi_left, dphi_left, -1.0),
        (nx - 1, phi_right, dphi_right, 1.0),
    ):
        trace, derivative = _face_basis(
            phi, endpoint, deriv_endpoint, "x", 2.0 / hx, outward,
        )
        cons_b = -measure_x * torch.einsum(
            "q,qk,ql->kl", weights, trace, derivative,
        )
        block = cons_b + cons_b.T + pen_x * measure_x * torch.einsum(
            "q,qk,ql->kl", weights, trace, trace,
        )
        for iy in range(ny):
            out[ix, iy] += block @ coeff[ix, iy]

    # Boundary faces (y)
    for iy, endpoint, deriv_endpoint, outward in (
        (0, phi_left, dphi_left, -1.0),
        (ny - 1, phi_right, dphi_right, 1.0),
    ):
        trace, derivative = _face_basis(
            phi, endpoint, deriv_endpoint, "y", 2.0 / hy, outward,
        )
        cons_b = -measure_y * torch.einsum(
            "q,qk,ql->kl", weights, trace, derivative,
        )
        block = cons_b + cons_b.T + pen_y * measure_y * torch.einsum(
            "q,qk,ql->kl", weights, trace, trace,
        )
        for ix in range(nx):
            out[ix, iy] += block @ coeff[ix, iy]

    return out.reshape(-1)


def solve_poisson_2d_matrixfree(
    ncells: int | tuple[int, int] = 16,
    degree: int = 1,
    penalty: float = 10.0,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    rtol: float = 1e-12,
    maxiter: int = 10_000,
) -> tuple:
    """Matrix-free CG solve for 2-D SIPG Poisson."""
    from tensordg.poisson_2d import _face_basis, _operators

    nx, ny = (ncells, ncells) if isinstance(ncells, int) else ncells
    dev = torch.device(device)
    hx, hy = 1.0 / nx, 1.0 / ny
    x = (torch.arange(nx, device=dev, dtype=dtype) + 0.5) * hx
    y = (torch.arange(ny, device=dev, dtype=dtype) + 0.5) * hy

    operators = _operators(degree, dev, dtype)
    nodes, weights, phi, mass, stiffness, phi_left, phi_right, dphi_left, dphi_right = operators
    modes = degree + 1
    local_dofs = modes * modes
    ndofs = nx * ny * local_dofs

    # Build RHS
    xq = x[:, None, None, None] + 0.5 * hx * nodes[None, None, :, None]
    yq = y[None, :, None, None] + 0.5 * hy * nodes[None, None, None, :]
    forcing = 2.0 * math.pi ** 2 * torch.sin(math.pi * xq) * torch.sin(math.pi * yq)
    load = 0.25 * hx * hy * torch.einsum(
        "xyqr,q,r,qk,rl->xykl", forcing, weights, weights, phi, phi,
    )
    rhs = load.reshape(-1)

    # Build diagonal preconditioner: compute the diagonal of the global operator
    # Each element's diagonal block gets contributions from:
    # 1. Volume integral
    # 2. Interior face contributions (LL/RR blocks for vertical, BB/TT for horizontal)
    # 3. Boundary face contributions (if on boundary)
    volume_x = (hy / hx) * torch.einsum(
        "ki,lj->klij", stiffness, mass,
    ).reshape(local_dofs, local_dofs)
    volume_y = (hx / hy) * torch.einsum(
        "ki,lj->klij", mass, stiffness,
    ).reshape(local_dofs, local_dofs)
    volume = volume_x + volume_y
    
    # Define face measures and penalties for boundary contributions
    measure_x = 0.5 * hy
    pen_x = penalty * modes * modes / hx
    measure_y = 0.5 * hx
    pen_y = penalty * modes * modes / hy
    
    # Interior face blocks (LL, RR for vertical; BB, TT for horizontal)
    # These contribute to the diagonal of the global matrix
    trace_right, deriv_right_f = _face_basis(
        phi, phi_right, dphi_right, "x", 2.0 / hx,
    )
    trace_left, deriv_left_f = _face_basis(
        phi, phi_left, dphi_left, "x", 2.0 / hx,
    )
    jump_x = torch.cat((trace_right, -trace_left), dim=1)
    avg_deriv_x = 0.5 * torch.cat((deriv_right_f, deriv_left_f), dim=1)
    cons_x = -measure_x * torch.einsum(
        "q,qk,ql->kl", weights, jump_x, avg_deriv_x,
    )
    face_x = cons_x + cons_x.T + pen_x * measure_x * torch.einsum(
        "q,qk,ql->kl", weights, jump_x, jump_x,
    )
    fLL_x = face_x[:local_dofs, :local_dofs]
    fRR_x = face_x[local_dofs:, local_dofs:]
    
    trace_top, deriv_top_f = _face_basis(
        phi, phi_right, dphi_right, "y", 2.0 / hy,
    )
    trace_bot, deriv_bot_f = _face_basis(
        phi, phi_left, dphi_left, "y", 2.0 / hy,
    )
    jump_y = torch.cat((trace_top, -trace_bot), dim=1)
    avg_deriv_y = 0.5 * torch.cat((deriv_top_f, deriv_bot_f), dim=1)
    cons_y = -measure_y * torch.einsum(
        "q,qk,ql->kl", weights, jump_y, avg_deriv_y,
    )
    face_y = cons_y + cons_y.T + pen_y * measure_y * torch.einsum(
        "q,qk,ql->kl", weights, jump_y, jump_y,
    )
    fBB_y = face_y[:local_dofs, :local_dofs]
    fTT_y = face_y[local_dofs:, local_dofs:]
    
    diag = torch.zeros(ndofs, device=dev, dtype=dtype)
    for ix in range(nx):
        for iy in range(ny):
            idx_start = (ix * ny + iy) * local_dofs
            block_diag = torch.diag(volume).clone()
            
            # Interior vertical face contributions
            if ix > 0:
                block_diag += torch.diag(fRR_x)  # right face of left neighbor
            if ix < nx - 1:
                block_diag += torch.diag(fLL_x)  # left face of right neighbor
            
            # Interior horizontal face contributions
            if iy > 0:
                block_diag += torch.diag(fTT_y)  # top face of bottom neighbor
            if iy < ny - 1:
                block_diag += torch.diag(fBB_y)  # bottom face of top neighbor
            
            diag[idx_start:idx_start + local_dofs] = block_diag

    # Add boundary face contributions to diagonal
    for ix, endpoint, deriv_endpoint, outward in (
        (0, phi_left, dphi_left, -1.0),
        (nx - 1, phi_right, dphi_right, 1.0),
    ):
        trace, derivative = _face_basis(
            phi, endpoint, deriv_endpoint, "x", 2.0 / hx, outward,
        )
        cons_b = -measure_x * torch.einsum(
            "q,qk,ql->kl", weights, trace, derivative,
        )
        block = cons_b + cons_b.T + pen_x * measure_x * torch.einsum(
            "q,qk,ql->kl", weights, trace, trace,
        )
        block_diag = torch.diag(block)
        for iy in range(ny):
            idx_start = (ix * ny + iy) * local_dofs
            diag[idx_start:idx_start + local_dofs] += block_diag

    for iy, endpoint, deriv_endpoint, outward in (
        (0, phi_left, dphi_left, -1.0),
        (ny - 1, phi_right, dphi_right, 1.0),
    ):
        trace, derivative = _face_basis(
            phi, endpoint, deriv_endpoint, "y", 2.0 / hy, outward,
        )
        cons_b = -measure_y * torch.einsum(
            "q,qk,ql->kl", weights, trace, derivative,
        )
        block = cons_b + cons_b.T + pen_y * measure_y * torch.einsum(
            "q,qk,ql->kl", weights, trace, trace,
        )
        block_diag = torch.diag(block)
        for ix in range(nx):
            idx_start = (ix * ny + iy) * local_dofs
            diag[idx_start:idx_start + local_dofs] += block_diag

    # Invert diagonal (with safeguard for near-zero entries)
    diag_safe = torch.where(
        torch.abs(diag) > 1e-14, diag, torch.ones_like(diag) * 1e-14
    )
    diag_inv = 1.0 / diag_safe

    def precond(v):
        return diag_inv * v

    def mv(v):
        return matvec_poisson_2d(v, nx, ny, degree, penalty, operators)

    solution, cg_info = _pcg(mv, rhs, precond=precond, rtol=rtol, maxiter=maxiter)
    coeff = solution.reshape(nx, ny, modes, modes)

    metadata = {
        "ncells": (nx, ny), "degree": degree, "dofs": ndofs,
        "penalty": penalty, "device": str(dev), "backend": "matrix-free",
        "cg_iterations": cg_info["iterations"],
        "cg_residual_norm": cg_info["residual_norm"],
    }
    return x, y, coeff, metadata


# ---------------------------------------------------------------------------
# 1-D Heat matrix-free
# ---------------------------------------------------------------------------

def matvec_heat_diffusion_1d(
    coeff_flat: torch.Tensor,
    ncells: int,
    degree: int,
    penalty: float,
    operators: tuple,
) -> torch.Tensor:
    """Compute the SIPG diffusion operator A @ coeff for 1-D heat."""
    h = 1.0 / ncells
    (
        nodes, weights, phi, mass, stiffness, phi_left, phi_right,
        dphi_left, dphi_right,
    ) = operators
    modes = degree + 1
    coeff = coeff_flat.reshape(ncells, modes)
    out = torch.zeros_like(coeff)

    # Volume
    volume = (2.0 / h) * stiffness
    for cell in range(ncells):
        out[cell] += volume @ coeff[cell]

    # Interior faces
    jump = torch.cat((phi_right, -phi_left))
    avg_deriv = 0.5 * (2.0 / h) * torch.cat((dphi_right, dphi_left))
    consistency = -torch.outer(jump, avg_deriv)
    face = (
        consistency + consistency.T
        + penalty * modes * modes / h * torch.outer(jump, jump)
    )
    fLL = face[:modes, :modes]
    fLR = face[:modes, modes:]
    fRL = face[modes:, :modes]
    fRR = face[modes:, modes:]
    for cell in range(ncells - 1):
        uL = coeff[cell]
        uR = coeff[cell + 1]
        out[cell] += fLL @ uL + fLR @ uR
        out[cell + 1] += fRL @ uL + fRR @ uR

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
        out[cell] += block @ coeff[cell]

    return out.reshape(-1)


def matvec_heat_mass_1d(
    coeff_flat: torch.Tensor,
    ncells: int,
    degree: int,
    operators: tuple,
) -> torch.Tensor:
    """Compute M @ coeff for 1-D heat (block-diagonal mass matrix)."""
    h = 1.0 / ncells
    nodes, weights, phi, mass, stiffness, *_ = operators
    modes = degree + 1
    volume_mass = 0.5 * h * mass
    coeff = coeff_flat.reshape(ncells, modes)
    out = torch.zeros_like(coeff)
    for cell in range(ncells):
        out[cell] = volume_mass @ coeff[cell]
    return out.reshape(-1)


def solve_heat_1d_matrixfree(
    ncells: int = 16,
    degree: int = 2,
    final_time: float = 0.1,
    time_steps: int | None = None,
    penalty: float = 20.0,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    rtol: float = 1e-12,
    maxiter: int = 5_000,
) -> tuple:
    """Matrix-free CN heat solve: each time step uses PCG."""
    from tensordg.heat_1d import _operators

    dev = torch.device(device)
    h = 1.0 / ncells
    centers = (torch.arange(ncells, device=dev, dtype=dtype) + 0.5) * h

    operators = _operators(degree, dev, dtype)
    nodes, weights, phi, mass, stiffness, phi_left, phi_right, dphi_left, dphi_right = operators
    modes = degree + 1
    ndofs = ncells * modes

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

    state = coeff.reshape(-1)

    # Build diagonal preconditioner for system = M + 0.5*dt*D
    volume_mass = 0.5 * h * mass
    volume_diff = (2.0 / h) * stiffness
    mass_diag = torch.diag(volume_mass)
    diff_diag = torch.diag(volume_diff)

    diag = torch.zeros(ndofs, device=dev, dtype=dtype)
    for cell in range(ncells):
        diag[cell * modes:(cell + 1) * modes] = mass_diag + 0.5 * dt * diff_diag

    # Add boundary face contributions
    for cell, endpoint, deriv_endpoint, outward in (
        (0, phi_left, dphi_left, -1.0),
        (ncells - 1, phi_right, dphi_right, 1.0),
    ):
        derivative = outward * (2.0 / h) * deriv_endpoint
        consistency_b = -torch.outer(endpoint, derivative)
        face_block = (
            consistency_b + consistency_b.T
            + penalty * modes * modes / h * torch.outer(endpoint, endpoint)
        )
        face_diag = torch.diag(face_block)
        diag[cell * modes:(cell + 1) * modes] += 0.5 * dt * face_diag

    diag_inv = torch.where(
        torch.abs(diag) > 1e-14, 1.0 / diag, torch.zeros_like(diag)
    )

    def precond(v):
        return diag_inv * v

    total_cg_iters = 0
    if steps:
        def system_mv(v):
            return (
                matvec_heat_mass_1d(v, ncells, degree, operators)
                + 0.5 * dt * matvec_heat_diffusion_1d(
                    v, ncells, degree, penalty, operators,
                )
            )

        def explicit_mv(v):
            return (
                matvec_heat_mass_1d(v, ncells, degree, operators)
                - 0.5 * dt * matvec_heat_diffusion_1d(
                    v, ncells, degree, penalty, operators,
                )
            )

        for _ in range(steps):
            b = explicit_mv(state)
            state, cg_info = _pcg(
                system_mv, b, precond=precond, rtol=rtol, maxiter=maxiter,
            )
            total_cg_iters += cg_info["iterations"]

    coeff = state.reshape(ncells, modes)
    metadata = {
        "ncells": ncells, "degree": degree, "final_time": final_time,
        "steps": steps, "dt": dt, "penalty": penalty, "dofs": ndofs,
        "device": str(dev), "backend": "matrix-free",
        "total_cg_iterations": total_cg_iters,
    }
    return centers, coeff, metadata
