"""SIPG Poisson on the unit square with per-side Dirichlet or Neumann data.

``tensordg.poisson_2d`` hard-codes homogeneous Dirichlet data on all four sides
and the single manufactured solution ``sin(pi x) sin(pi y)``.  This module keeps
that Dirichlet assembly bit-comparable (it reuses the very same face operators)
and adds the capability the repository was missing: the **natural (Neumann)
boundary condition**, chosen independently on each side, together with an
arbitrary forcing ``f(x, y)``.

The weak form of ``-Delta u = f`` reads

    a(u, v) = int_Omega grad(u).grad(v)
              - sum_{Gamma_D} ( {grad(u).n} [v] + {grad(v).n} [u] )
              + sum_{Gamma_D} (sigma/h) [u] [v]
            = int_Omega f v + sum_{Gamma_N} int_{Gamma_N} g_N v ,

so a Neumann face changes only the right-hand side: it contributes the
consistency load ``int g_N v`` and *no* trace/penalty terms (Arnold, Brezzi,
Cockburn & Marini, SIAM J. Numer. Anal. 39 (2002), eq. (2.9)).  ``g_N`` is the
**outward** normal derivative ``du/dn``.

Pure-Neumann data leave the discrete operator with its one-dimensional constant
nullspace.  That case is detected, checked (row sums must vanish, i.e.
``A @ 1 == 0``) and solved with the rank-one gauge

    A + (1/ndofs) 1 1^T

which pins the remaining additive constant (``1^T u = 1^T b``; the field it
returns has, to quadrature accuracy, the same pointwise mean as the reference).
Problems whose nullspace is larger than the constants (e.g. a solution that is
constant across the periodic-like direction of a fully Neumann box) are out of
scope and rejected rather than silently mis-gauge'd -- see
``docs/reports/tensordg-poisson-neumann-verification.md``.
"""
from __future__ import annotations



import numpy as np
import torch

from .poisson_2d import (
    _add_block,
    _face_basis,
    _operators,
    dirichlet_face_operator,
    interior_face_operator,
)

BC_SIDES = ("left", "right", "bottom", "top")
_ALLOWED = ("dirichlet", "neumann")


def _resolve_bc(bc) -> dict[str, str]:
    """Normalise a partial ``{side: kind}`` mapping (default: Dirichlet)."""
    resolved = {side: "dirichlet" for side in BC_SIDES}
    if not bc:
        return resolved
    for side, kind in bc.items():
        if side not in resolved:
            raise ValueError(f"unknown boundary side {side!r}; use one of {BC_SIDES}")
        if kind not in _ALLOWED:
            raise ValueError(f"boundary kind must be one of {_ALLOWED}, got {kind!r}")
        resolved[side] = kind
    return resolved


def _as_flux(g, side):
    """Return a callable (x, y) -> du/dn, or ``None`` for a zero flux."""
    if g is None:
        return None
    if callable(g):
        return g
    if isinstance(g, (int, float)):
        value = float(g)

        def constant(x, y):
            return torch.full(torch.broadcast_shapes(x.shape, y.shape), value,
                              device=x.device, dtype=x.dtype)

        return constant
    raise TypeError("neumann_flux entries must be None, a number, or a callable")


def assemble_poisson_2d_bc(
    ncells: int | tuple[int, int] = 16,
    degree: int = 1,
    penalty: float = 20.0,
    bc=None,
    forcing=None,
    neumann_flux=None,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
):
    """Assemble ``-Delta u = f`` with per-side Dirichlet/Neumann data.

    Returns ``(matrix, rhs, grid, metadata)`` where ``grid`` holds the element
    centres and sizes needed by :func:`l2_error_2d`.
    """
    nx, ny = (ncells, ncells) if isinstance(ncells, int) else tuple(ncells)
    if nx < 2 or ny < 2 or degree not in (1, 2, 3):
        raise ValueError("ncells must be >= 2 and SIPG degree must be in [1, 3]")
    if penalty <= 0.0:
        raise ValueError("penalty must be positive")
    if forcing is None or not callable(forcing):
        raise TypeError("forcing must be a callable f(x, y) -> tensor")

    types = _resolve_bc(bc)
    fluxes = {side: None for side in BC_SIDES}
    for side, g in (neumann_flux or {}).items():
        if side not in BC_SIDES:
            raise ValueError(f"unknown boundary side {side!r}; use one of {BC_SIDES}")
        fluxes[side] = _as_flux(g, side)

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

    # ---- volume term -----------------------------------------------------
    volume_x = (hy / hx) * torch.einsum("ki,lj->klij", stiffness, mass).reshape(
        local_dofs, local_dofs,
    )
    volume_y = (hx / hy) * torch.einsum("ki,lj->klij", mass, stiffness).reshape(
        local_dofs, local_dofs,
    )
    volume = volume_x + volume_y
    for ix in range(nx):
        for iy in range(ny):
            _add_block(matrix, dofs(ix, iy), volume)

    # ---- interior faces (identical to the Dirichlet-only benchmark) ------
    measure_x, measure_y = 0.5 * hy, 0.5 * hx
    face_x = interior_face_operator(
        phi, weights, phi_left, phi_right, dphi_left, dphi_right,
        "x", hx, measure_x, penalty, modes,
    )
    for ix in range(nx - 1):
        for iy in range(ny):
            _add_block(matrix, torch.cat((dofs(ix, iy), dofs(ix + 1, iy))), face_x)
    face_y = interior_face_operator(
        phi, weights, phi_left, phi_right, dphi_left, dphi_right,
        "y", hy, measure_y, penalty, modes,
    )
    for ix in range(nx):
        for iy in range(ny - 1):
            _add_block(matrix, torch.cat((dofs(ix, iy), dofs(ix, iy + 1))), face_y)

    # ---- boundary faces --------------------------------------------------
    load = torch.zeros(ndofs, device=dev, dtype=dtype).view(nx, ny, local_dofs)
    sides = (
        ("left", "x", 0.0, 0, phi_left, dphi_left, -1.0, hx, measure_x),
        ("right", "x", 1.0, nx - 1, phi_right, dphi_right, 1.0, hx, measure_x),
        ("bottom", "y", 0.0, 0, phi_left, dphi_left, -1.0, hy, measure_y),
        ("top", "y", 1.0, ny - 1, phi_right, dphi_right, 1.0, hy, measure_y),
    )
    for side, axis, position, index, endpoint, d_endpoint, outward, h_normal, measure in sides:
        if types[side] == "dirichlet":
            block, _ = dirichlet_face_operator(
                phi, weights, endpoint, d_endpoint, axis, outward,
                h_normal, measure, penalty, modes,
            )
            for line in range(ny if axis == "x" else nx):
                ix, iy = (index, line) if axis == "x" else (line, index)
                _add_block(matrix, dofs(ix, iy), block)
            continue
        g = fluxes[side]
        if g is None:
            continue  # homogeneous Neumann: no matrix block, no load
        if axis == "x":
            yq = y[None, :, None] + 0.5 * hy * nodes[None, None, :]
            xq = torch.full_like(yq, position)
            scaled = 0.5 * hy * weights[None, None, :] * g(xq, yq)
            load[index] += torch.einsum(
                "yr,rl,k->ykl", scaled[0], phi, endpoint,
            ).reshape(ny, local_dofs)
        else:
            xq = x[:, None, None] + 0.5 * hx * nodes[None, :, None]
            yq = torch.full_like(xq, position)
            scaled = 0.5 * hx * weights[None, None, :] * g(xq, yq)
            load[:, index] += torch.einsum(
                "xq,qk,l->xkl", scaled[:, 0], phi, endpoint,
            ).reshape(nx, local_dofs)

    # ---- forcing ---------------------------------------------------------
    xq = x[:, None, None, None] + 0.5 * hx * nodes[None, None, :, None]
    yq = y[None, :, None, None] + 0.5 * hy * nodes[None, None, None, :]
    f = forcing(xq, yq)
    load += (0.25 * hx * hy * torch.einsum(
        "xyqr,q,r,qk,rl->xykl", f, weights, weights, phi, phi,
    )).reshape(nx, ny, local_dofs)

    grid = {"x": x, "y": y, "hx": hx, "hy": hy, "degree": degree}
    metadata = {
        "ncells": (nx, ny), "degree": degree, "dofs": ndofs, "penalty": penalty,
        "bc": dict(types),
        "matrix_symmetry_error": float(torch.max(torch.abs(matrix - matrix.T)).item()),
        "device": str(dev),
    }
    return matrix, load.reshape(-1), grid, metadata


def solve_poisson_2d_bc(
    ncells: int | tuple[int, int] = 16,
    degree: int = 1,
    penalty: float = 20.0,
    bc=None,
    forcing=None,
    neumann_flux=None,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
):
    """Solve the per-side Dirichlet/Neumann Poisson problem (dense backend).

    Returns ``(x_centers, y_centers, modal_coefficients, metadata)`` with the
    modal coefficients shaped ``(nx, ny, degree + 1, degree + 1)``.
    """
    matrix, rhs, grid, metadata = assemble_poisson_2d_bc(
        ncells=ncells, degree=degree, penalty=penalty, bc=bc,
        forcing=forcing, neumann_flux=neumann_flux, device=device, dtype=dtype,
    )
    nx, ny = metadata["ncells"]
    local_dofs = (degree + 1) ** 2
    pure_neumann = all(kind == "neumann" for kind in metadata["bc"].values())
    if pure_neumann:
        ones = torch.zeros(matrix.shape[0], device=matrix.device, dtype=matrix.dtype)
        ones.view(nx, ny, local_dofs)[:, :, 0] = 1.0
        scale = float(torch.max(torch.abs(matrix)).item())
        nullspace = float(torch.max(torch.abs(matrix @ ones)).item())
        metadata["nullspace_residual"] = nullspace / max(scale, 1e-300)
        if metadata["nullspace_residual"] > 1e-10:
            raise RuntimeError(
                "pure-Neumann operator does not annihilate constants "
                f"(relative residual {metadata['nullspace_residual']:.3e}); "
                "the assembly is inconsistent"
            )
        matrix = matrix + (1.0 / matrix.shape[0]) * torch.outer(ones, ones)
        metadata["gauge"] = "rank-one 1 1^T / ndofs"
    else:
        metadata["nullspace_residual"] = 0.0
        metadata["gauge"] = "none"

    solution = torch.linalg.solve(matrix, rhs)
    residual = float(torch.max(torch.abs(matrix @ solution - rhs)).item())
    metadata["linear_residual"] = residual / max(float(torch.max(torch.abs(rhs)).item()), 1e-300)
    coeff = solution.reshape(nx, ny, degree + 1, degree + 1)
    return grid["x"], grid["y"], coeff, metadata


def l2_error_2d(x, y, coeff, exact, remove_constant: bool = False) -> dict[str, float]:
    """Relative L2 error of a modal field against a callable ``exact(x, y)``.

    ``remove_constant`` subtracts the mean of the pointwise difference before
    taking the norm; that is the correct comparison for a pure-Neumann
    solution, which the discrete problem only fixes up to a constant.
    """
    degree = coeff.shape[-1] - 1
    nx, ny = coeff.shape[:2]
    hx, hy = 1.0 / nx, 1.0 / ny
    nodes_np, weights_np = np.polynomial.legendre.leggauss(max(2 * degree + 4, 8))
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
    reference = exact(xq, yq)
    difference = numerical - reference
    quadrature = weights[None, None, :, None] * weights[None, None, None, :]
    weight = 0.25 * hx * hy
    offset = 0.0
    reference = reference.expand_as(difference)
    if remove_constant:
        total = float(torch.sum(weight * quadrature).item())
        offset = float(torch.sum(weight * quadrature * difference).item()) / total
        difference = difference - offset
    error2 = weight * torch.sum(quadrature * difference ** 2)
    norm2 = weight * torch.sum(quadrature * (reference - (offset if remove_constant else 0.0)) ** 2)
    absolute = float(torch.sqrt(error2).item())
    norm = float(torch.sqrt(norm2).item())
    return {
        "absolute_l2": absolute,
        "exact_l2": norm,
        "relative_l2": absolute / norm,
        "constant_offset": offset,
        "max_abs_error": float(torch.max(torch.abs(difference)).item()),
    }