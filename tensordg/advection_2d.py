"""Two-dimensional periodic linear advection with tensor-product DG.

The discretization uses modal Legendre polynomials on a uniform Cartesian
mesh, an upwind numerical flux, Gauss--Legendre projection, and explicit Runge--Kutta time stepping (RK4 by default).
The exact solution is a periodically translated smooth trigonometric profile.
"""
from __future__ import annotations

from collections.abc import Callable
import math

import numpy as np
import torch

from .runge_kutta import get_tableau
from ._advection_time import advance_transport
from .advection_1d import _copy_modal_state, _departure_coordinate

# The tensor-product operators are built from Legendre values/derivatives at
# Gauss nodes, so they are degree-generic; the bound only keeps the Python-side
# basis assembly sane (see ``tensordg.advection_1d.MAX_DEGREE``).
MAX_DEGREE = 8


def _domain_lengths(lengths):
    sides = tuple(float(v) for v in lengths)
    if len(sides) != 2 or any(not math.isfinite(v) or v <= 0 for v in sides):
        raise ValueError("lengths must contain two finite positive box lengths")
    return sides


def _sample_profile(profile, x, y, lengths):
    if profile is None:
        return 1.0 + torch.sin(2.0 * math.pi * x / lengths[0]) * torch.sin(2.0 * math.pi * y / lengths[1])
    x, y = torch.broadcast_tensors(x, y)
    values = torch.as_tensor(profile(x, y), device=x.device, dtype=x.dtype)
    return torch.broadcast_to(values, x.shape)


def _operators(degree: int, device: torch.device, dtype: torch.dtype):
    """Return one-dimensional reference operators reused in both directions."""
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
    # V[k,i] = integral P_i(xi) d(P_k)/dxi dxi.
    volume = torch.einsum("q,qi,qk->ki", weights, phi, dphi)
    phi_left = torch.tensor(
        [(-1.0) ** k for k in range(degree + 1)],
        device=device, dtype=dtype,
    )
    phi_right = torch.ones(degree + 1, device=device, dtype=dtype)
    return nodes, weights, phi, volume, phi_left, phi_right


def _rhs(
    u: torch.Tensor,
    hx: float,
    hy: float,
    velocity: tuple[float, float],
    volume: torch.Tensor,
    phi_left: torch.Tensor,
    phi_right: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the semi-discrete DG operator for ``u[cell_x,cell_y,k,l]``."""
    ax, ay = velocity

    # x-directed volume and upwind face terms. Y modes are spectators.
    volume_x = torch.einsum("xyil,ki->xykl", u, volume)
    right_x = torch.einsum("xyil,i->xyl", u, phi_right)
    left_x = torch.einsum("xyil,i->xyl", u, phi_left)
    if ax >= 0.0:
        flux_left_x = torch.roll(right_x, shifts=1, dims=0)
        flux_right_x = right_x
    else:
        flux_left_x = left_x
        flux_right_x = torch.roll(left_x, shifts=-1, dims=0)

    # y-directed volume and upwind face terms. X modes are spectators.
    volume_y = torch.einsum("xykj,lj->xykl", u, volume)
    top_y = torch.einsum("xykj,j->xyk", u, phi_right)
    bottom_y = torch.einsum("xykj,j->xyk", u, phi_left)
    if ay >= 0.0:
        # The upwind state is the upper trace of the cell below each face.
        flux_bottom_y = torch.roll(top_y, shifts=1, dims=1)
        flux_top_y = top_y
    else:
        flux_bottom_y = bottom_y
        flux_top_y = torch.roll(bottom_y, shifts=-1, dims=1)

    out = torch.zeros_like(u)
    for k in range(u.shape[2]):
        out[:, :, k, :] += ((2 * k + 1) * ax / hx) * (
            volume_x[:, :, k, :]
            - flux_right_x * phi_right[k]
            + flux_left_x * phi_left[k]
        )
    for l in range(u.shape[3]):
        out[:, :, :, l] += ((2 * l + 1) * ay / hy) * (
            volume_y[:, :, :, l]
            - flux_top_y * phi_right[l]
            + flux_bottom_y * phi_left[l]
        )
    return out


from .registry import capability
@capability(
    name="advection-2d",
    category="solver",
    maturity="planned",
    evidence={"feature_ledger": "CASE-DG-SOLID-BODY-ROTATION-2D"},
    description="2D advection DG (solid-body rotation / deformational flow)",
    tags=['advection', '2d', 'dg']
)
def solve_advection_2d(
    ncells: int | tuple[int, int] = 24,
    degree: int = 1,
    final_time: float = 1.0,
    velocity: tuple[float, float] | Callable[[float], tuple[float, float]] = (1.0, 0.5),
    cfl: float = 0.08,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    *,
    initial: Callable | torch.Tensor | None = None,
    start_time: float = 0.0,
    time_integrator: str = "rk4",
    max_speed: tuple[float, float] | None = None,
    max_dt: float | None = None,
    velocity_jump_times: tuple[float, ...] = (),
    lengths: tuple[float, float] = (1.0, 1.0),
):
    """Solve ``u_t + ax*u_x + ay*u_y = 0`` on a periodic rectangle with side lengths (Lx, Ly).

    The initial/exact profile is ``1 + sin(2*pi*x/Lx) * sin(2*pi*y/Ly)`` translated
    by the time integral of the spatially uniform velocity. Supply ``initial(x, y)`` to project another periodic profile. ``ncells`` can be one integer or ``(nx,ny)``.

    A tensor passed as initial contains modal coefficients on the same mesh and
    physical rectangle at start_time. final_time is the absolute end time.

    A velocity(t) callback receives absolute stage times. Supply max_speed
    for componentwise bounds, max_dt to resolve rapid variation and
    velocity_jump_times for known right-continuous interior switches.
    time_integrator selects a registered explicit Runge--Kutta tableau.

    Returns ``(x_centers, y_centers, modal_coefficients, metadata)``. The
    coefficient array has shape ``(nx,ny,degree+1,degree+1)``; its ``[0,0]``
    mode is the cell average.
    """
    tableau = get_tableau(time_integrator)
    nx, ny = (ncells, ncells) if isinstance(ncells, int) else ncells
    if nx < 2 or ny < 2 or not 0 <= degree <= MAX_DEGREE:
        raise ValueError(
            f"ncells must be >= 2 and degree must be in [0, {MAX_DEGREE}]"
        )
    duration = final_time - start_time
    if not math.isfinite(duration) or duration < 0.:
        raise ValueError("start_time and final_time must define a finite forward interval")
    if not math.isfinite(cfl) or cfl <= 0.:
        raise ValueError("cfl must be positive and finite")

    dev = torch.device(device)
    lengths = _domain_lengths(lengths)
    hx, hy = lengths[0] / nx, lengths[1] / ny
    x = (torch.arange(nx, device=dev, dtype=dtype) + 0.5) * hx
    y = (torch.arange(ny, device=dev, dtype=dtype) + 0.5) * hy
    nodes, weights, phi, volume, phi_left, phi_right = _operators(
        degree, dev, dtype,
    )

    if isinstance(initial, torch.Tensor):
        coeff = _copy_modal_state(initial, (nx, ny, degree+1, degree+1), dev, dtype)
    else:
        xq = x[:, None, None, None] + 0.5 * hx * nodes[None, None, :, None]
        yq = y[None, :, None, None] + 0.5 * hy * nodes[None, None, None, :]
        samples = _sample_profile(initial, xq, yq, lengths)
        coeff = torch.einsum(
            "xyqr,q,r,qk,rl->xykl", samples, weights, weights, phi, phi,
        )
        norm = torch.tensor(
            [(2 * k + 1) * (2 * l + 1) / 4.0
             for k in range(degree + 1) for l in range(degree + 1)],
            device=dev, dtype=dtype,
        ).reshape(degree + 1, degree + 1)
        coeff = coeff * norm[None, None, :, :]

    def spatial_rhs(state, components):
        return _rhs(state, hx, hy, components, volume, phi_left, phi_right)

    coeff, velocity_info, bounds, steps, dt = advance_transport(
        coeff, spatial_rhs, (hx, hy), degree, tableau, velocity, start_time, final_time,
        cfl, max_speed, max_dt, velocity_jump_times)
    metadata = {
        "ncells": (nx, ny), "lengths": lengths, "degree": degree, "final_time": final_time,
        "velocity": velocity_info, "max_speed": bounds, "start_time": start_time, "dt": dt,
        "time_integrator": tableau.name, "rk_order": tableau.order, "rk_stages": tableau.stages,
        "steps": steps, "device": str(dev), "cfl": cfl,
    }
    return x, y, coeff, metadata


def error_2d(
    x: torch.Tensor,
    y: torch.Tensor,
    coeff: torch.Tensor,
    final_time: float,
    velocity: tuple[float, float] = (1.0, 0.5),
    *,
    initial: Callable | None = None,
    lengths: tuple[float, float] = (1.0, 1.0),
) -> dict[str, float]:
    """Compare against initial(x, y) translated along both periodic characteristics.

    Use the same physical lengths as the solver. Absolute L2 integrates over the rectangle.
    For a zero reference norm, relative L2 is zero for exact agreement and infinity otherwise.
    """
    degree = coeff.shape[-1] - 1
    nx, ny = coeff.shape[:2]
    lengths = _domain_lengths(lengths)
    hx, hy = lengths[0] / nx, lengths[1] / ny
    nodes_np, weights_np = np.polynomial.legendre.leggauss(
        max(2 * degree + 4, 6),
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
    ax, ay = velocity
    exact = _sample_profile(initial, _departure_coordinate(xq, final_time, ax, lengths[0]),
                            _departure_coordinate(yq, final_time, ay, lengths[1]), lengths)
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
        "relative_l2": absolute / norm if norm else (0.0 if absolute == 0.0 else math.inf),
    }
