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

which fixes the spatial mean to zero for a compatible right-hand side. The
constant vector in the modal basis has only its cell-mean entries set to one.
The assembled source and outward normal derivative must sum to zero when
tested against this vector.
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


def _boundary_samples(data, side, centers, nodes):
    """One row per face, one column per tangential quadrature point."""
    along = centers[:, None] + nodes[None, :]/(2*len(centers))
    fixed = torch.full_like(along, float(side in ("right", "top")))
    x, y = (fixed, along) if side in ("left", "right") else (along, fixed)
    values = data(x, y) if callable(data) else data
    return torch.broadcast_to(torch.as_tensor(values, device=along.device, dtype=along.dtype), along.shape)


def assemble_poisson_2d_bc(
    ncells: int | tuple[int, int] = 16,
    degree: int = 1,
    penalty: float = 20.0,
    bc=None,
    forcing=None,
    neumann_flux=None,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    *, dirichlet_values=None,
):
    """Assemble ``-Delta u = f`` with per-side Dirichlet/Neumann data.

    dirichlet_values maps prescribed sides to scalar values or callables of
    (x, y). Unspecified Dirichlet traces are zero. neumann_flux supplies du/dn.
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
    fluxes = neumann_flux or {}
    traces = dirichlet_values or {}
    for side in (*fluxes, *traces):
        if side not in BC_SIDES:
            raise ValueError(f"unknown boundary side {side!r}; use one of {BC_SIDES}")
    for side in traces:
        if types[side] != "dirichlet":
            raise ValueError(f"Dirichlet trace supplied on {types[side]} side {side}")

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
            block, trace = dirichlet_face_operator(
                phi, weights, endpoint, d_endpoint, axis, outward,
                h_normal, measure, penalty, modes,
            )
            for line in range(ny if axis == "x" else nx):
                ix, iy = (index, line) if axis == "x" else (line, index)
                _add_block(matrix, dofs(ix, iy), block)
            data = traces.get(side)
            if data is None:
                continue
            _, derivative = _face_basis(phi, endpoint, d_endpoint, axis, 2/h_normal, outward)
            test = penalty*modes**2/h_normal*trace - derivative
        else:
            data = fluxes.get(side)
            if data is None:
                continue
            test, _ = _face_basis(phi, endpoint, d_endpoint, axis, 2/h_normal, outward)
        samples = _boundary_samples(data, side, y if axis == "x" else x, nodes)
        contribution = measure*((samples*weights) @ test)
        if axis == "x":
            load[index] += contribution
        else:
            load[:,index] += contribution

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
    *, dirichlet_values=None,
):
    """Solve the per-side Dirichlet/Neumann Poisson problem (dense backend).

    Returns ``(x_centers, y_centers, modal_coefficients, metadata)`` with the
    modal coefficients shaped ``(nx, ny, degree + 1, degree + 1)``.
    """
    matrix, rhs, grid, metadata = assemble_poisson_2d_bc(
        ncells=ncells, degree=degree, penalty=penalty, bc=bc,
        forcing=forcing, neumann_flux=neumann_flux, device=device, dtype=dtype,
        dirichlet_values=dirichlet_values,
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
        means = rhs.view(nx, ny, local_dofs)[:, :, 0]
        imbalance = float(means.sum().item())
        metadata["net_neumann_load"] = imbalance
        # Scale by the full assembled load to include moments of nonconstant
        # data when the cell-mean components nearly cancel.
        tolerance = 128*torch.finfo(rhs.dtype).eps*float(rhs.abs().sum().item())
        if abs(imbalance) > tolerance:
            raise ValueError("incompatible pure Neumann data: integrated source plus "
                             f"outward normal derivative is {imbalance:.6g} (tolerance {tolerance:.3g})")
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


def poisson_face_powers_2d(coeff, *, penalty=20., bc=None, dirichlet_values=None, neumann_flux=None):
    """Integrate SIPG numerical heat flux over each face for unit conductivity and depth.

    Returns arrays (nx+1, ny) and (nx, ny+1) oriented in increasing x and y.
    Supply the same penalty and boundary data as the solve. For each cell,
    right minus left plus top minus bottom equals its integrated source.
    """
    nx, ny, modes, last = coeff.shape
    if modes != last or modes not in (2, 3, 4) or min(nx, ny) < 2 or penalty <= 0:
        raise ValueError("expected square P1 to P3 modal blocks and positive penalty")
    nodes, weights, phi, _, _, pl, pr, dl, dr = _operators(modes-1, coeff.device, coeff.dtype)
    types = _resolve_bc(bc)
    fields = coeff.flatten(2)
    outputs = []
    for axis, count, tangential, low, high in (("x", nx, ny, "left", "right"),
                                               ("y", ny, nx, "bottom", "top")):
        u = fields if axis == "x" else fields.transpose(0, 1)
        tl, gl = _face_basis(phi, pl, dl, axis, 2*count)
        tr, gr = _face_basis(phi, pr, dr, axis, 2*count)
        scale = penalty*modes*modes*count
        interior = -.5*(u[:-1] @ gr.T + u[1:] @ gl.T) + scale*(u[:-1] @ tr.T-u[1:] @ tl.T)
        result = coeff.new_empty((count+1, tangential))
        result[1:-1] = (interior @ weights)/(2*tangential)
        centers = (torch.arange(tangential, device=coeff.device, dtype=coeff.dtype)+.5)/tangential
        for side, index, sign, trace, gradient in ((low, 0, -1, tl, gl), (high, -1, 1, tr, gr)):
            neumann = types[side] == "neumann"
            data = ((neumann_flux if neumann else dirichlet_values) or {}).get(side)
            prescribed = torch.zeros_like(result[index, :, None]) if data is None else _boundary_samples(
                data, side, centers, nodes)
            if neumann:
                flux = -sign*prescribed.expand(tangential, len(nodes))
            else:
                flux = -(u[index] @ gradient.T) + sign*scale*(u[index] @ trace.T-prescribed)
            result[index] = (flux @ weights)/(2*tangential)
        outputs.append(result if axis == "x" else result.T)
    return tuple(outputs)


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
        total = nx * ny * float(torch.sum(weight * quadrature).item())
        offset = float(torch.sum(weight * quadrature * difference).item()) / total
        difference = difference - offset
    error2 = weight * torch.sum(quadrature * difference ** 2)
    if remove_constant:
        reference = reference - (weight * torch.sum(quadrature * reference))/total
    norm2 = weight * torch.sum(quadrature * reference**2)
    absolute = float(torch.sqrt(error2).item())
    norm = float(torch.sqrt(norm2).item())
    return {
        "absolute_l2": absolute,
        "exact_l2": norm,
        "relative_l2": absolute / norm,
        "constant_offset": offset,
        "max_abs_error": float(torch.max(torch.abs(difference)).item()),
    }