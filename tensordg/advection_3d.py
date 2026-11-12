"""Three-dimensional periodic linear advection with tensor-product DG.

This is the exact 3-D analogue of :mod:`tensordg.advection_2d`: modal Legendre
polynomials on a uniform Cartesian hexahedral mesh, an upwind numerical flux,
Gauss--Legendre projection of the initial condition, and classical RK4 in time.
It extends the shipped verification surface from 1-D/2-D Cartesian meshes to
3-D, which is the mesh topology of every production DG solver.

The initial/exact profile is the separable trigonometric field

    u0(x, y, z) = 1 + sin(2*pi*x) * sin(2*pi*y) * sin(2*pi*z),

translated rigidly by ``velocity * t`` on the periodic unit cube, so the exact
solution at any time is available in closed form:

    u(x, y, z, t) = 1 + sin(2*pi*(x - ax*t))
                        * sin(2*pi*(y - ay*t))
                        * sin(2*pi*(z - az*t)).

The unit offset gives the field a non-zero, analytically known mass (``1``),
which makes the conservation check non-trivial: the ``[0, 0, 0]`` coefficient
is the cell average and the global integral telescopes to round-off for any
upwind velocity, including sign-changing components.

Normalisation follows the lower-dimensional modules: mode ``[0, 0, 0]`` is the
cell average. With ``x = x_c + (h_x/2) xi`` (and likewise in y, z) the modal
semi-discrete operator is

    dc_klm/dt = (2k+1)(2l+1)(2m+1)/(8 h_x h_y h_z) *
                ( int_K (a u) . grad(phi_k phi_l phi_m) dV
                  - int_dK (a.n) uh phi_k phi_l phi_m dS ),

which is what :func:`_rhs` implements direction by direction; the Jacobian
factors collapse to the familiar ``(2k+1) a_x / h_x`` per axis.
"""
from __future__ import annotations

import math

import numpy as np
import torch

Velocity3 = tuple[float, float, float]


def _operators(degree: int, device: torch.device, dtype: torch.dtype):
    """Return the 1-D reference operators reused along the three axes."""
    nq = max(2 * degree + 3, 4)
    nodes_np, weights_np = np.polynomial.legendre.leggauss(nq)
    nodes = torch.as_tensor(nodes_np, device=device, dtype=dtype)
    weights = torch.as_tensor(weights_np, device=device, dtype=dtype)
    values, derivatives = [], []
    for k in range(degree + 1):
        polynomial = np.zeros(k + 1)
        polynomial[-1] = 1.0
        derivative = np.polynomial.legendre.legder(polynomial)
        values.append(torch.as_tensor(
            np.polynomial.legendre.legval(nodes_np, polynomial),
            device=device, dtype=dtype,
        ))
        derivative_values = (
            np.polynomial.legendre.legval(nodes_np, derivative)
            if len(derivative) else np.zeros_like(nodes_np)
        )
        derivatives.append(torch.as_tensor(
            derivative_values, device=device, dtype=dtype,
        ))
    phi = torch.stack(values, dim=1)
    dphi = torch.stack(derivatives, dim=1)
    # V[k, i] = integral P_i(xi) d(P_k)/dxi dxi over the reference element.
    volume = torch.einsum("q,qi,qk->ki", weights, phi, dphi)
    phi_left = torch.tensor(
        [(-1.0) ** k for k in range(degree + 1)],
        device=device, dtype=dtype,
    )
    phi_right = torch.ones(degree + 1, device=device, dtype=dtype)
    return nodes, weights, phi, volume, phi_left, phi_right


def _rhs(
    u: torch.Tensor,
    spacing: Velocity3,
    velocity: Velocity3,
    volume: torch.Tensor,
    phi_left: torch.Tensor,
    phi_right: torch.Tensor,
) -> torch.Tensor:
    """Semi-discrete DG operator for ``u[cell_x, cell_y, cell_z, k, l, m]``."""
    hx, hy, hz = spacing
    ax, ay, az = velocity
    nk, nl, nm = u.shape[3], u.shape[4], u.shape[5]

    out = torch.zeros_like(u)

    # --- x direction: the (l, m) modes are spectators ---------------------
    volume_x = torch.einsum("xyzilm,ki->xyzklm", u, volume)
    right_x = torch.einsum("xyzilm,i->xyzlm", u, phi_right)
    left_x = torch.einsum("xyzilm,i->xyzlm", u, phi_left)
    if ax >= 0.0:
        flux_left_x = torch.roll(right_x, shifts=1, dims=0)
        flux_right_x = right_x
    else:
        flux_left_x = left_x
        flux_right_x = torch.roll(left_x, shifts=-1, dims=0)
    for k in range(nk):
        out[:, :, :, k, :, :] += ((2 * k + 1) * ax / hx) * (
            volume_x[:, :, :, k, :, :]
            - flux_right_x * phi_right[k]
            + flux_left_x * phi_left[k]
        )

    # --- y direction: the (k, m) modes are spectators ---------------------
    volume_y = torch.einsum("xyzkjm,lj->xyzklm", u, volume)
    top_y = torch.einsum("xyzkjm,j->xyzkm", u, phi_right)
    bottom_y = torch.einsum("xyzkjm,j->xyzkm", u, phi_left)
    if ay >= 0.0:
        flux_bottom_y = torch.roll(top_y, shifts=1, dims=1)
        flux_top_y = top_y
    else:
        flux_bottom_y = bottom_y
        flux_top_y = torch.roll(bottom_y, shifts=-1, dims=1)
    for l in range(nl):
        out[:, :, :, :, l, :] += ((2 * l + 1) * ay / hy) * (
            volume_y[:, :, :, :, l, :]
            - flux_top_y * phi_right[l]
            + flux_bottom_y * phi_left[l]
        )

    # --- z direction: the (k, l) modes are spectators ---------------------
    volume_z = torch.einsum("xyzklj,mj->xyzklm", u, volume)
    top_z = torch.einsum("xyzklj,j->xyzkl", u, phi_right)
    bottom_z = torch.einsum("xyzklj,j->xyzkl", u, phi_left)
    if az >= 0.0:
        flux_bottom_z = torch.roll(top_z, shifts=1, dims=2)
        flux_top_z = top_z
    else:
        flux_bottom_z = bottom_z
        flux_top_z = torch.roll(bottom_z, shifts=-1, dims=2)
    for m in range(nm):
        out[:, :, :, :, :, m] += ((2 * m + 1) * az / hz) * (
            volume_z[:, :, :, :, :, m]
            - flux_top_z * phi_right[m]
            + flux_bottom_z * phi_left[m]
        )
    return out


def _project_initial(
    nx: int, ny: int, nz: int,
    degree: int,
    spacing: Velocity3,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Modal coefficients of ``1 + sin(2 pi x) sin(2 pi y) sin(2 pi z)``.

    The initial field is separable, so the 3-D tensor product of the three 1-D
    projections is exact (up to the Gauss quadrature of the sine modes) and
    costs ``O(N (p+1))`` instead of ``O((N (p+1))**3)``.
    """
    hx, hy, hz = spacing
    nq = max(2 * degree + 3, 4)
    nodes_np, weights_np = np.polynomial.legendre.leggauss(nq)
    nodes = torch.as_tensor(nodes_np, device=device, dtype=dtype)
    weights = torch.as_tensor(weights_np, device=device, dtype=dtype)
    basis = []
    for k in range(degree + 1):
        polynomial = np.zeros(k + 1)
        polynomial[-1] = 1.0
        basis.append(torch.as_tensor(
            np.polynomial.legendre.legval(nodes_np, polynomial),
            device=device, dtype=dtype,
        ))
    phi = torch.stack(basis, dim=1)

    def _axis_modes(ncells: int, h: float) -> torch.Tensor:
        centres = (torch.arange(ncells, device=device, dtype=dtype) + 0.5) * h
        xq = centres[:, None] + 0.5 * h * nodes[None, :]
        samples = torch.sin(2.0 * math.pi * xq)
        # (2k+1)/2 * sum_q w_q f_q P_k(q)  -- per cell.
        modes = torch.einsum("q,cq,qk->ck", weights, samples, phi) * 0.5
        scales = torch.tensor(
            [2 * k + 1 for k in range(degree + 1)],
            device=device, dtype=dtype,
        )
        return modes * scales[None, :]

    mx = _axis_modes(nx, hx)
    my = _axis_modes(ny, hy)
    mz = _axis_modes(nz, hz)
    coeff = torch.einsum("xk,yl,zm->xyzklm", mx, my, mz)
    # The constant part of the field only projects onto the cell-average mode.
    coeff[:, :, :, 0, 0, 0] += 1.0
    return coeff


def exact_solution_3d(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    final_time: float,
    velocity: Velocity3 = (1.0, 0.5, 0.25),
) -> torch.Tensor:
    """Closed-form solution of the periodic 3-D advection test."""
    ax, ay, az = velocity
    return 1.0 + (
        torch.sin(2.0 * math.pi * (x - ax * final_time))
        * torch.sin(2.0 * math.pi * (y - ay * final_time))
        * torch.sin(2.0 * math.pi * (z - az * final_time))
    )


def solve_advection_3d(
    ncells: int | tuple[int, int, int] = 8,
    degree: int = 1,
    final_time: float = 1.0,
    velocity: Velocity3 = (1.0, 0.5, 0.25),
    cfl: float = 0.08,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
):
    """Solve ``u_t + a . grad u = 0`` on the periodic unit cube.

    ``ncells`` is one integer or ``(nx, ny, nz)``. Returns
    ``(x, y, z, modal_coefficients, metadata)`` where ``x``, ``y``, ``z`` are
    the cell centres and the coefficient array has shape
    ``(nx, ny, nz, degree+1, degree+1, degree+1)`` whose ``[0, 0, 0]`` entry is
    the cell average.
    """
    nx, ny, nz = (
        (ncells, ncells, ncells) if isinstance(ncells, int) else tuple(ncells)
    )
    if min(nx, ny, nz) < 2 or degree not in (0, 1, 2, 3):
        raise ValueError("ncells must be >= 2 and degree must be in [0, 3]")
    if final_time < 0.0 or cfl <= 0.0:
        raise ValueError("final_time must be >= 0 and cfl must be > 0")
    if len(velocity) != 3:
        raise ValueError("velocity must contain exactly three components")

    dev = torch.device(device)
    hx, hy, hz = 1.0 / nx, 1.0 / ny, 1.0 / nz
    x = (torch.arange(nx, device=dev, dtype=dtype) + 0.5) * hx
    y = (torch.arange(ny, device=dev, dtype=dtype) + 0.5) * hy
    z = (torch.arange(nz, device=dev, dtype=dtype) + 0.5) * hz
    _, _, _, volume, phi_left, phi_right = _operators(degree, dev, dtype)

    coeff = _project_initial(nx, ny, nz, degree, (hx, hy, hz), dev, dtype)

    ax, ay, az = (float(component) for component in velocity)
    wave_rate = abs(ax) / hx + abs(ay) / hy + abs(az) / hz
    if final_time == 0.0 or wave_rate == 0.0:
        steps = 0
        dt_nominal = float("inf")
    else:
        dt_nominal = cfl / ((2 * degree + 1) * wave_rate)
        steps = math.ceil(final_time / dt_nominal)
    dt = final_time / steps if steps else 0.0

    spacing = (hx, hy, hz)
    velocity3 = (ax, ay, az)
    for _ in range(steps):
        k1 = _rhs(coeff, spacing, velocity3, volume, phi_left, phi_right)
        k2 = _rhs(
            coeff + 0.5 * dt * k1, spacing, velocity3,
            volume, phi_left, phi_right,
        )
        k3 = _rhs(
            coeff + 0.5 * dt * k2, spacing, velocity3,
            volume, phi_left, phi_right,
        )
        k4 = _rhs(
            coeff + dt * k3, spacing, velocity3,
            volume, phi_left, phi_right,
        )
        coeff = coeff + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    metadata = {
        "ncells": (nx, ny, nz), "degree": degree, "final_time": final_time,
        "velocity": velocity3, "dt": dt, "steps": steps,
        "device": str(dev), "cfl": cfl, "dtype": str(dtype),
    }
    return x, y, z, coeff, metadata


def mass_3d(coeff: torch.Tensor) -> float:
    """Total integral of the modal field (``sum`` of cell averages / N)."""
    nx, ny, nz = coeff.shape[:3]
    return float(coeff[:, :, :, 0, 0, 0].sum().item()) / (nx * ny * nz)


def error_3d(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    coeff: torch.Tensor,
    final_time: float,
    velocity: Velocity3 = (1.0, 0.5, 0.25),
    x_chunk: int = 4,
) -> dict[str, float]:
    """Quadrature-based absolute/relative L2 errors against the closed form.

    The quadrature field ``u_h - u_exact`` would be an ``(nx*nq)**3`` tensor, so
    the norm is accumulated in ``x``-slabs to keep the memory footprint bounded
    for the finest meshes.
    """
    degree = coeff.shape[-1] - 1
    nx, ny, nz = coeff.shape[:3]
    hx, hy, hz = 1.0 / nx, 1.0 / ny, 1.0 / nz
    nq = max(2 * degree + 4, 6)
    nodes_np, weights_np = np.polynomial.legendre.leggauss(nq)
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

    xq = x[:, None] + 0.5 * hx * nodes[None, :]
    yq = y[:, None] + 0.5 * hy * nodes[None, :]
    zq = z[:, None] + 0.5 * hz * nodes[None, :]
    ax, ay, az = velocity
    sin_x = torch.sin(2.0 * math.pi * (xq - ax * final_time))
    sin_y = torch.sin(2.0 * math.pi * (yq - ay * final_time))
    sin_z = torch.sin(2.0 * math.pi * (zq - az * final_time))

    # (1, 1, 1, nq, nq, nq) tensor-product quadrature weights, matching the
    # (cell_x, cell_y, cell_z, qx, qy, qz) layout of ``numerical`` below.
    quad_w = (
        weights[None, None, None, :, None, None]
        * weights[None, None, None, None, :, None]
        * weights[None, None, None, None, None, :]
    )
    cell_volume = hx * hy * hz / 8.0
    error2 = coeff.new_zeros(())
    norm2 = coeff.new_zeros(())
    for start in range(0, nx, x_chunk):
        stop = min(start + x_chunk, nx)
        slab = coeff[start:stop]
        numerical = torch.einsum("xyzklm,qk->xyzqlm", slab, phi)
        numerical = torch.einsum("xyzqlm,rl->xyzqrm", numerical, phi)
        numerical = torch.einsum("xyzqrm,sm->xyzqrs", numerical, phi)
        exact = 1.0 + (
            sin_x[start:stop, None, None, :, None, None]
            * sin_y[None, :, None, None, :, None]
            * sin_z[None, None, :, None, None, :]
        )
        error2 = error2 + cell_volume * torch.sum(quad_w * (numerical - exact) ** 2)
        norm2 = norm2 + cell_volume * torch.sum(quad_w * exact ** 2)
    absolute = float(torch.sqrt(error2).item())
    norm = float(torch.sqrt(norm2).item())
    return {
        "absolute_l2": absolute,
        "exact_l2": norm,
        "relative_l2": absolute / norm,
    }