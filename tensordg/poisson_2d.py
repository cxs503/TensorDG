"""Two-dimensional Poisson benchmark using a symmetric interior-penalty DG method.

The model problem is -Delta(u) = f on the unit square with homogeneous
Dirichlet boundary data and the manufactured exact solution
u = sin(pi*x) sin(pi*y). A dense torch solve is intentional: it keeps this
small verification benchmark independent of a sparse-linear-algebra package.

Three backends are supported via the ``backend`` parameter:
  * ``"dense"``        -- assemble a dense torch matrix, solve with torch.linalg.solve
  * ``"sparse"``       -- assemble a scipy CSR matrix, solve with scipy.sparse.linalg.spsolve
  * ``"matrix-free"``  -- no global assembly, solve with preconditioned CG
"""
from __future__ import annotations

import math

import numpy as np
import torch


def _operators(degree: int, device: torch.device, dtype: torch.dtype):
    nq = max(2 * degree + 3, 6)
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
            np.polynomial.legendre.legval(nodes_np, polynomial),
            device=device, dtype=dtype,
        ))
        deriv_values = (
            np.polynomial.legendre.legval(nodes_np, derivative)
            if len(derivative) else np.zeros_like(nodes_np)
        )
        derivatives.append(torch.as_tensor(
            deriv_values, device=device, dtype=dtype,
        ))
        left.append((-1.0) ** k)
        right.append(1.0)
        if len(derivative):
            deriv_left.append(float(np.polynomial.legendre.legval(-1.0, derivative)))
            deriv_right.append(float(np.polynomial.legendre.legval(1.0, derivative)))
        else:
            deriv_left.append(0.0)
            deriv_right.append(0.0)
    phi = torch.stack(values, dim=1)
    dphi = torch.stack(derivatives, dim=1)
    mass = torch.einsum("q,qk,qi->ki", weights, phi, phi)
    stiffness = torch.einsum("q,qk,qi->ki", weights, dphi, dphi)
    tensor = lambda data: torch.as_tensor(data, device=device, dtype=dtype)
    return (
        nodes, weights, phi, mass, stiffness, tensor(left), tensor(right),
        tensor(deriv_left), tensor(deriv_right),
    )


def _face_basis(
    phi: torch.Tensor,
    endpoint: torch.Tensor,
    derivative_endpoint: torch.Tensor,
    direction: str,
    derivative_scale: float,
    normal_sign: float = 1.0,
):
    """Construct trace and normal-derivative matrices on one element face."""
    modes = phi.shape[1]
    local_dofs = modes * modes
    trace = torch.zeros(
        (phi.shape[0], local_dofs), device=phi.device, dtype=phi.dtype,
    )
    derivative = torch.zeros_like(trace)
    for i in range(modes):
        for j in range(modes):
            column = i * modes + j
            if direction == "x":
                trace[:, column] = endpoint[i] * phi[:, j]
                derivative[:, column] = (
                    normal_sign * derivative_scale
                    * derivative_endpoint[i] * phi[:, j]
                )
            else:
                trace[:, column] = phi[:, i] * endpoint[j]
                derivative[:, column] = (
                    normal_sign * derivative_scale
                    * phi[:, i] * derivative_endpoint[j]
                )
    return trace, derivative


def _add_block(matrix: torch.Tensor, indices: torch.Tensor, block: torch.Tensor):
    matrix[indices[:, None], indices[None, :]] += block


def solve_poisson_2d(
    ncells: int | tuple[int, int] = 16,
    degree: int = 1,
    penalty: float = 20.0,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    backend: str = "dense",
):
    """Solve the manufactured Poisson problem with SIPG on a Cartesian mesh.

    Returns ``(x_centers, y_centers, modal_coefficients, metadata)``. The
    penalty parameter is scaled by ``(degree+1)**2 / face_height``.

    Parameters
    ----------
    backend : str
        - ``"dense"``: assemble dense torch matrix (default, backward compatible)
        - ``"sparse"``: assemble scipy CSR matrix, solve with spsolve
        - ``"matrix-free"``: no global assembly, solve with PCG
    """
    if backend not in ("dense", "sparse", "matrix-free"):
        raise ValueError("backend must be 'dense', 'sparse', or 'matrix-free'")

    if backend == "sparse":
        from tensordg.sipg_sparse import solve_poisson_2d_sparse
        return solve_poisson_2d_sparse(ncells, degree, penalty, device, dtype)
    elif backend == "matrix-free":
        from tensordg.sipg_matrixfree import solve_poisson_2d_matrixfree
        return solve_poisson_2d_matrixfree(ncells, degree, penalty, device, dtype)

    # backend == "dense": original implementation
    nx, ny = (ncells, ncells) if isinstance(ncells, int) else ncells
    if nx < 2 or ny < 2 or degree not in (1, 2, 3):
        raise ValueError("ncells must be >= 2 and SIPG degree must be in [1, 3]")
    if penalty <= 0.0:
        raise ValueError("penalty must be positive")

    dev = torch.device(device)
    hx, hy = 1.0 / nx, 1.0 / ny
    x = (torch.arange(nx, device=dev, dtype=dtype) + 0.5) * hx
    y = (torch.arange(ny, device=dev, dtype=dtype) + 0.5) * hy
    (
        nodes, weights, phi, mass, stiffness, phi_left, phi_right,
        dphi_left, dphi_right,
    ) = _operators(degree, dev, dtype)
    modes = degree + 1
    local_dofs = modes * modes
    ndofs = nx * ny * local_dofs
    matrix = torch.zeros((ndofs, ndofs), device=dev, dtype=dtype)
    local = torch.arange(local_dofs, device=dev)

    def dofs(ix: int, iy: int):
        return (ix * ny + iy) * local_dofs + local

    # Volume: integral grad(u).grad(v), split into tensor-product x/y terms.
    volume_x = (hy / hx) * torch.einsum(
        "ki,lj->klij", stiffness, mass,
    ).reshape(local_dofs, local_dofs)
    volume_y = (hx / hy) * torch.einsum(
        "ki,lj->klij", mass, stiffness,
    ).reshape(local_dofs, local_dofs)
    volume = volume_x + volume_y
    for ix in range(nx):
        for iy in range(ny):
            _add_block(matrix, dofs(ix, iy), volume)

    # Interior vertical faces. The global face normal points from left to
    # right; jump = left-right and average = 1/2(left+right).
    trace_right, deriv_right = _face_basis(
        phi, phi_right, dphi_right, "x", 2.0 / hx,
    )
    trace_left, deriv_left = _face_basis(
        phi, phi_left, dphi_left, "x", 2.0 / hx,
    )
    jump_x = torch.cat((trace_right, -trace_left), dim=1)
    average_derivative_x = 0.5 * torch.cat((deriv_right, deriv_left), dim=1)
    measure_x = 0.5 * hy
    consistency_x = -measure_x * torch.einsum(
        "q,qk,ql->kl", weights, jump_x, average_derivative_x,
    )
    penalty_x = penalty * modes * modes / hx
    face_x = consistency_x + consistency_x.T + penalty_x * measure_x * torch.einsum(
        "q,qk,ql->kl", weights, jump_x, jump_x,
    )
    for ix in range(nx - 1):
        for iy in range(ny):
            indices = torch.cat((dofs(ix, iy), dofs(ix + 1, iy)))
            _add_block(matrix, indices, face_x)

    # Interior horizontal faces, with global normal pointing bottom to top.
    trace_top, deriv_top = _face_basis(
        phi, phi_right, dphi_right, "y", 2.0 / hy,
    )
    trace_bottom, deriv_bottom = _face_basis(
        phi, phi_left, dphi_left, "y", 2.0 / hy,
    )
    jump_y = torch.cat((trace_top, -trace_bottom), dim=1)
    average_derivative_y = 0.5 * torch.cat((deriv_top, deriv_bottom), dim=1)
    measure_y = 0.5 * hx
    consistency_y = -measure_y * torch.einsum(
        "q,qk,ql->kl", weights, jump_y, average_derivative_y,
    )
    penalty_y = penalty * modes * modes / hy
    face_y = consistency_y + consistency_y.T + penalty_y * measure_y * torch.einsum(
        "q,qk,ql->kl", weights, jump_y, jump_y,
    )
    for ix in range(nx):
        for iy in range(ny - 1):
            indices = torch.cat((dofs(ix, iy), dofs(ix, iy + 1)))
            _add_block(matrix, indices, face_y)

    # Homogeneous Dirichlet boundaries use the same SIPG terms with the
    # exterior trace equal to zero and no averaging factor.
    for ix, endpoint, deriv_endpoint, outward in (
        (0, phi_left, dphi_left, -1.0),
        (nx - 1, phi_right, dphi_right, 1.0),
    ):
        trace, derivative = _face_basis(
            phi, endpoint, deriv_endpoint, "x", 2.0 / hx, outward,
        )
        consistency = -measure_x * torch.einsum(
            "q,qk,ql->kl", weights, trace, derivative,
        )
        block = consistency + consistency.T + penalty_x * measure_x * torch.einsum(
            "q,qk,ql->kl", weights, trace, trace,
        )
        for iy in range(ny):
            _add_block(matrix, dofs(ix, iy), block)

    for iy, endpoint, deriv_endpoint, outward in (
        (0, phi_left, dphi_left, -1.0),
        (ny - 1, phi_right, dphi_right, 1.0),
    ):
        trace, derivative = _face_basis(
            phi, endpoint, deriv_endpoint, "y", 2.0 / hy, outward,
        )
        consistency = -measure_y * torch.einsum(
            "q,qk,ql->kl", weights, trace, derivative,
        )
        block = consistency + consistency.T + penalty_y * measure_y * torch.einsum(
            "q,qk,ql->kl", weights, trace, trace,
        )
        for ix in range(nx):
            _add_block(matrix, dofs(ix, iy), block)

    # Manufactured forcing f = 2*pi^2*sin(pi*x)*sin(pi*y).
    xq = x[:, None, None, None] + 0.5 * hx * nodes[None, None, :, None]
    yq = y[None, :, None, None] + 0.5 * hy * nodes[None, None, None, :]
    forcing = 2.0 * math.pi ** 2 * torch.sin(math.pi * xq) * torch.sin(math.pi * yq)
    load = 0.25 * hx * hy * torch.einsum(
        "xyqr,q,r,qk,rl->xykl", forcing, weights, weights, phi, phi,
    )
    rhs = load.reshape(-1)
    solution = torch.linalg.solve(matrix, rhs)
    coeff = solution.reshape(nx, ny, modes, modes)
    symmetry_error = float(torch.max(torch.abs(matrix - matrix.T)).item())
    metadata = {
        "ncells": (nx, ny), "degree": degree, "dofs": ndofs,
        "penalty": penalty, "matrix_symmetry_error": symmetry_error,
        "device": str(dev),
    }
    return x, y, coeff, metadata


def poisson_error_2d(
    x: torch.Tensor,
    y: torch.Tensor,
    coeff: torch.Tensor,
) -> dict[str, float]:
    """Return absolute and relative L2 errors against sin(pi*x)sin(pi*y)."""
    degree = coeff.shape[-1] - 1
    nx, ny = coeff.shape[:2]
    hx, hy = 1.0 / nx, 1.0 / ny
    nodes_np, weights_np = np.polynomial.legendre.leggauss(
        max(2 * degree + 4, 8),
    )
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
    numerical = torch.einsum("xykl,qk,rl->xyqr", coeff, phi, phi)
    xq = x[:, None, None, None] + 0.5 * hx * nodes[None, None, :, None]
    yq = y[None, :, None, None] + 0.5 * hy * nodes[None, None, None, :]
    exact = torch.sin(math.pi * xq) * torch.sin(math.pi * yq)
    error2 = 0.25 * hx * hy * torch.sum(
        weights[None, None, :, None] * weights[None, None, None, :]
        * (numerical - exact) ** 2,
    )
    norm2 = 0.25 * hx * hy * torch.sum(
        weights[None, None, :, None] * weights[None, None, None, :]
        * exact ** 2,
    )
    absolute = float(torch.sqrt(error2).item())
    norm = float(torch.sqrt(norm2).item())
    return {
        "absolute_l2": absolute,
        "exact_l2": norm,
        "relative_l2": absolute / norm,
    }
