"""High-order tensor-product modal DG for the two-dimensional Euler equations.

``euler_2d.py`` in this package is the P0 (finite-volume limit) baseline on a
periodic Cartesian mesh.  This module adds the genuinely high-order 2-D
capability: a tensor-product Legendre modal basis of arbitrary degree, an HLLC
normal-direction interface flux (Toro's three-wave solver, the same flux used by
the 1-D high-order solver), SSP-RK3 time stepping, a positivity-preserving
projection fallback and an optional directional minmod (TVB, Cockburn & Shu
1989) shock limiter.

Everything is written with plain PyTorch tensor algebra, so the whole residual
pipeline stays autograd-compatible.

Verification status (mission m2400, supersedes the m2143 honest FAIL)

* residual identity: a Sod tube extruded to 2-D (``ny = 1``, ``y``-invariant
  data) reproduces the *independently written* 1-D DG residual
  (``tensordg/euler_dg_1d.py``) to machine precision for every cell, mode and
  component -- this is what the m2143 defect broke;
* smooth isentropic vortex: design-order convergence ``EOC = degree + 1`` and
  L2 error below the 3 % gate at the documented resolution;
* Sod tube extruded to 2-D: agreement with the exact Riemann solution;
* genuinely 2-D four-quadrant Riemann problem (Lax & Liu 1998, config 3):
  positivity plus mesh self-convergence (no closed-form solution exists).

Reproducers: ``python -m benchmarks.verify_euler_dg_2d`` (numbers + JSON) and
``python -m benchmarks.diagnose_euler_dg_2d`` (the m2143 defect, now closed).

Defect history (m2143 -> m2400): ``_axis_fluxes`` built the right-hand
neighbour trace *without* rolling it in the non-periodic case, so HLLC received
the cell's own left trace as its right state and silently collapsed to the
physical flux.  The residual of the extruded Sod case was then identically
zero.  The roll is now unconditional and only the outermost face is
overwritten with the interior trace (outflow).
"""
from __future__ import annotations

import numpy as np
import torch

from .euler_1d import GAMMA
from .euler_2d import vortex_primitive
from .limiters import _weno_weights
from .tvb import troubled_from_differences

__all__ = [
    "solve_euler_dg_2d",
    "solve_euler_dg_2d_sod",
    "solve_euler_dg_2d_riemann",
    "solve_euler_dg_2d_vortex",
    "cell_fields",
    "legendre_operators",
    "conservative_state",
    "normalise_bcs",
]


# ---------------------------------------------------------------------------
# Basis / quadrature
# ---------------------------------------------------------------------------

def legendre_operators(degree: int, quadrature: int, device="cpu", dtype=torch.float64):
    """Return (nodes, weights, basis, derivative, signs) for a modal DG basis.

    The basis is the Legendre family ``P_m`` on ``[-1, 1]`` normalised by
    ``int_{-1}^{1} P_m P_n dx = 2 / (2m + 1) delta_mn``.
    """
    if degree < 0:
        raise ValueError("degree must be non-negative")
    if quadrature < degree + 1:
        raise ValueError("quadrature must be at least degree + 1")
    nodes_np, weights_np = np.polynomial.legendre.leggauss(quadrature)
    basis = np.polynomial.legendre.legvander(nodes_np, degree)  # [q, degree+1]
    derivative = np.zeros_like(basis)
    for mode in range(degree + 1):
        coefficients = np.zeros(mode + 1)
        coefficients[mode] = 1.0
        differentiated = np.polynomial.legendre.legder(coefficients)
        if differentiated.size:
            derivative[:, mode] = np.polynomial.legendre.legval(
                nodes_np, differentiated
            )
    # P_m(+1) = 1 and P_m(-1) = (-1)^m for every Legendre mode.
    signs = np.asarray([(-1.0) ** mode for mode in range(degree + 1)])
    return tuple(
        torch.as_tensor(array, device=device, dtype=dtype)
        for array in (nodes_np, weights_np, basis, derivative, signs)
    )


# ---------------------------------------------------------------------------
# Equation-of-state helpers (four-component 2-D state)
# ---------------------------------------------------------------------------

def _pressure(state: torch.Tensor) -> torch.Tensor:
    rho = state[..., 0].clamp_min(1e-12)
    return (GAMMA - 1.0) * (
        state[..., 3] - 0.5 * (state[..., 1] ** 2 + state[..., 2] ** 2) / rho
    )


def _primitive(state: torch.Tensor):
    rho = state[..., 0].clamp_min(1e-12)
    return rho, state[..., 1] / rho, state[..., 2] / rho, _pressure(state)


def _conservative(rho, velocity_x, velocity_y, pressure):
    energy = pressure / (GAMMA - 1.0) + 0.5 * rho * (
        velocity_x ** 2 + velocity_y ** 2
    )
    return torch.stack((rho, rho * velocity_x, rho * velocity_y, energy), dim=-1)


def _physical_flux(state: torch.Tensor, axis: int) -> torch.Tensor:
    rho, velocity_x, velocity_y, pressure = _primitive(state)
    if axis == 0:
        return torch.stack(
            (
                state[..., 1],
                state[..., 1] * velocity_x + pressure,
                state[..., 1] * velocity_y,
                velocity_x * (state[..., 3] + pressure),
            ),
            dim=-1,
        )
    return torch.stack(
        (
            state[..., 2],
            state[..., 2] * velocity_x,
            state[..., 2] * velocity_y + pressure,
            velocity_y * (state[..., 3] + pressure),
        ),
        dim=-1,
    )


def _hllc(ul: torch.Tensor, ur: torch.Tensor, axis: int) -> torch.Tensor:
    """Three-wave HLLC flux with a Rusanov fallback for degenerate data."""
    normal = 1 if axis == 0 else 2
    rho_l = ul[..., 0].clamp_min(1e-12)
    rho_r = ur[..., 0].clamp_min(1e-12)
    normal_l = ul[..., normal] / rho_l
    normal_r = ur[..., normal] / rho_r
    pressure_l = _pressure(ul).clamp_min(1e-12)
    pressure_r = _pressure(ur).clamp_min(1e-12)
    sound_l = torch.sqrt(GAMMA * pressure_l / rho_l)
    sound_r = torch.sqrt(GAMMA * pressure_r / rho_r)

    speed_l = torch.minimum(normal_l - sound_l, normal_r - sound_r)
    speed_r = torch.maximum(normal_l + sound_l, normal_r + sound_r)
    denominator = rho_l * (speed_l - normal_l) - rho_r * (speed_r - normal_r)
    safe_denominator = torch.where(
        denominator.abs() < 1e-30,
        torch.full_like(denominator, 1e-30),
        denominator,
    )
    speed_middle = (
        pressure_r
        - pressure_l
        + rho_l * normal_l * (speed_l - normal_l)
        - rho_r * normal_r * (speed_r - normal_r)
    ) / safe_denominator

    def star_state(state, rho, normal_velocity, pressure, speed):
        # ``speed - normal_velocity`` is legitimately negative for the left
        # acoustic wave (it equals the negative sound speed), so only the
        # magnitude may be guarded -- the sign must be preserved.
        gap = speed - normal_velocity
        gap = torch.where(
            gap.abs() < 1e-14,
            torch.where(gap < 0.0, -torch.full_like(gap, 1e-14), torch.full_like(gap, 1e-14)),
            gap,
        )
        rho_star = rho * (speed - normal_velocity) / (speed - speed_middle)
        energy_star = rho_star * (
            state[..., 3] / rho
            + (speed_middle - normal_velocity)
            * (speed_middle + pressure / (rho * gap))
        )
        momentum_normal = rho_star * speed_middle
        tangential_velocity = state[..., 2] / rho if axis == 0 else state[..., 1] / rho
        momentum_tangential = rho_star * tangential_velocity
        if axis == 0:
            momentum = torch.stack(
                (rho_star, momentum_normal, momentum_tangential, energy_star), dim=-1
            )
        else:
            momentum = torch.stack(
                (rho_star, momentum_tangential, momentum_normal, energy_star), dim=-1
            )
        return momentum

    flux_l = _physical_flux(ul, axis)
    flux_r = _physical_flux(ur, axis)
    star_l = star_state(ul, rho_l, normal_l, pressure_l, speed_l)
    star_r = star_state(ur, rho_r, normal_r, pressure_r, speed_r)
    star_flux_l = flux_l + speed_l[..., None] * (star_l - ul)
    star_flux_r = flux_r + speed_r[..., None] * (star_r - ur)

    flux = torch.where((speed_l >= 0.0)[..., None], flux_l, star_flux_l)
    flux = torch.where((speed_middle < 0.0)[..., None], star_flux_r, flux)
    flux = torch.where((speed_r <= 0.0)[..., None], flux_r, flux)

    alpha = torch.maximum(normal_l.abs() + sound_l, normal_r.abs() + sound_r)
    rusanov = 0.5 * (flux_l + flux_r) - 0.5 * alpha[..., None] * (ur - ul)
    degenerate = (speed_r - speed_l).abs() < 1e-13
    return torch.where(degenerate[..., None], rusanov, flux)


# ---------------------------------------------------------------------------
# Spatial operator
# ---------------------------------------------------------------------------

def _traces(coeff, basis, signs, axis):
    """Traces at ``xi = +1`` and ``xi = -1`` along ``axis``.

    The trace keeps the *surface* quadrature index of the remaining direction
    so the interface flux can be evaluated pointwise: index ``b`` (y) for
    x-faces and index ``a`` (x) for y-faces.  ``P_m(+1) = 1`` is used
    implicitly, ``P_m(-1) = (-1)**m`` is applied through ``signs``.
    """
    if axis == 0:
        right = torch.einsum("xyklv,bl->xybv", coeff, basis)
        left = torch.einsum("xyklv,k,bl->xybv", coeff, signs, basis)
    else:
        right = torch.einsum("xyklv,ak->xyav", coeff, basis)
        left = torch.einsum("xyklv,l,ak->xyav", coeff, signs, basis)
    return right, left


# --- m16803 boundary conditions ---
#
# ``solve_euler_dg_2d`` historically exposed only periodic faces and a
# zero-gradient ("outflow") outer face, which made the Woodward--Colella double
# Mach reflection impossible: the reflecting floor and the tracked post-shock
# inflow have no representation.  The boundary is now described per face as
#
#   * ``"outflow"`` / ``None`` -- interior trace on both sides (old behaviour),
#   * ``"wall"`` -- impermeable inviscid slip wall, implemented with the exact
#     wall flux ``(0, p, 0, 0)`` normal to the face (no mass, momentum or
#     energy leak; free-stream preservation is exact, not O(M^2)),
#   * ``callable(X, Y, t) -> state`` or ``-> (state, wall_mask)`` -- a general
#     ghost state given at the face quadrature points ``X, Y`` (physical
#     coordinates, shapes ``(n_face, q)``) and the stage time ``t``.  ``state``
#     is a *conservative* ``(rho, rho u, rho v, E)`` array broadcastable to
#     ``(n_face, q, 4)``; a ``True`` entry in ``wall_mask`` switches that point
#     to the exact wall flux.  ``conservative_state`` builds the array from
#     primitive variables.  This covers supersonic inflow, tracked moving
#     post-shock states and piecewise wall/inflow segments on one face.


def conservative_state(rho, velocity_x, velocity_y, pressure):
    """Conservative variables ``(rho, rho u, rho v, E)`` from primitives."""
    return _conservative(rho, velocity_x, velocity_y, pressure)


def _wall_flux(state, axis):
    """Exact inviscid slip-wall flux: only the normal pressure does work."""
    pressure = _pressure(state).clamp_min(0.0)
    flux = torch.zeros_like(state)
    flux[..., 1 if axis == 0 else 2] = pressure
    return flux


def _face_points(geom, axis, high):
    """Physical coordinates of the quadrature points of a boundary face."""
    nodes = geom["nodes"]
    count = nodes.numel()
    if axis == 0:
        x_face = geom["xmax"] if high else geom["xmin"]
        X = torch.full((geom["ny"], count), x_face, device=nodes.device, dtype=nodes.dtype)
        Y = geom["y"][:, None] + 0.5 * geom["dy"] * nodes[None, :]
    else:
        y_face = geom["ymax"] if high else geom["ymin"]
        X = geom["x"][:, None] + 0.5 * geom["dx"] * nodes[None, :]
        Y = torch.full((geom["nx"], count), y_face, device=nodes.device, dtype=nodes.dtype)
    return X, Y


def _ghost_state(bc, interior, X, Y, time, axis):
    """Exterior trace of a boundary face and its optional wall mask."""
    if bc is None or bc == "outflow":
        return interior, None
    if bc == "wall":
        ghost = interior.clone()
        ghost[..., 1 if axis == 0 else 2] = -ghost[..., 1 if axis == 0 else 2]
        return ghost, torch.ones(interior.shape[:-1], device=interior.device, dtype=torch.bool)
    if bc == "periodic":  # pragma: no cover - guarded in solve_euler_dg_2d
        raise ValueError("a periodic axis must not carry a boundary condition")
    out = bc(X, Y, time)
    wall = None
    if isinstance(out, tuple):
        out, wall = out
    state = torch.as_tensor(out, device=interior.device, dtype=interior.dtype)
    state = torch.broadcast_to(state, interior.shape)
    if wall is not None:
        wall = torch.as_tensor(wall, device=interior.device, dtype=torch.bool)
        wall = torch.broadcast_to(wall, interior.shape[:-1])
    return state, wall


def _boundary_flux(bc, interior, weights, basis, geom, axis, high, time):
    """Integrated numerical flux through one boundary face."""
    if bc is None or bc == "outflow":
        # Zero-gradient outer face: both traces are the interior one, so HLLC
        # collapses to the physical flux.  No quadrature coordinates needed,
        # which keeps the residual callable without a geometry record.
        flux = _hllc(interior, interior, axis)
        if axis == 0:
            return torch.einsum("ybv,b,bl->ylv", flux, weights, basis)
        return torch.einsum("xav,a,ak->xkv", flux, weights, basis)
    X, Y = _face_points(geom, axis, high)
    ghost, wall = _ghost_state(bc, interior, X, Y, time, axis)
    if high:
        flux = _hllc(interior, ghost, axis)
    else:
        flux = _hllc(ghost, interior, axis)
    if wall is not None:
        flux = torch.where(wall.unsqueeze(-1), _wall_flux(interior, axis), flux)
    if axis == 0:
        return torch.einsum("ybv,b,bl->ylv", flux, weights, basis)
    return torch.einsum("xav,a,ak->xkv", flux, weights, basis)


def _axis_fluxes(coeff, basis, signs, weights, periodic, axis, geom, stage_time=0.0):
    """Integrated interface fluxes ``(F_right, F_left)`` per cell.

    ``F_right[x, y, l, v]`` (x direction) is the numerical flux of the face
    between cell ``x`` and cell ``x+1``, projected onto the test functions of
    the remaining direction.  Non-periodic boundaries take their exterior trace
    from ``geom["bcs"]`` (default ``None`` = outflow, i.e. the interior trace on
    both sides, which makes HLLC collapse to the physical flux).
    """
    trace_right, trace_left = _traces(coeff, basis, signs, axis)
    if axis == 0:
        # ``neighbour[x]`` is the *left* trace of the cell to the right of
        # face ``x``; the roll is required for both the periodic and the
        # outflow case (m2400: omitting it made HLLC see the cell's own state
        # and silently degrade to the physical flux).
        neighbour = torch.roll(trace_left, shifts=-1, dims=0)
        if not periodic:
            neighbour[-1] = trace_right[-1]
        interface = _hllc(trace_right, neighbour, axis)
        flux_right = torch.einsum("xybv,b,bl->xylv", interface, weights, basis)
        flux_left = torch.roll(flux_right, shifts=1, dims=0)
        if not periodic:
            bcs = geom["bcs"] if geom else {}
            flux_left[0] = _boundary_flux(
                bcs.get("left"), trace_left[0], weights, basis, geom, 0, False, stage_time
            )
            flux_right[-1] = _boundary_flux(
                bcs.get("right"), trace_right[-1], weights, basis, geom, 0, True, stage_time
            )
    else:
        neighbour = torch.roll(trace_left, shifts=-1, dims=1)
        if not periodic:
            neighbour[:, -1] = trace_right[:, -1]
        interface = _hllc(trace_right, neighbour, axis)
        flux_right = torch.einsum("xyav,a,ak->xykv", interface, weights, basis)
        flux_left = torch.roll(flux_right, shifts=1, dims=1)
        if not periodic:
            bcs = geom["bcs"] if geom else {}
            flux_left[:, 0] = _boundary_flux(
                bcs.get("bottom"), trace_left[:, 0], weights, basis, geom, 1, False, stage_time
            )
            flux_right[:, -1] = _boundary_flux(
                bcs.get("top"), trace_right[:, -1], weights, basis, geom, 1, True, stage_time
            )
    return flux_right, flux_left


def normalise_bcs(bcs):
    """Validate and normalise the ``bcs`` mapping of ``solve_euler_dg_2d``."""
    if bcs is None:
        return {}
    if not isinstance(bcs, dict):
        raise ValueError("bcs must be a dict keyed by 'left'/'right'/'bottom'/'top'")
    allowed = {"left", "right", "bottom", "top"}
    unknown = set(bcs) - allowed
    if unknown:
        raise ValueError(f"unknown boundary faces {sorted(unknown)}")
    for face, value in bcs.items():
        if value is None or value in ("outflow", "wall"):
            continue
        if not callable(value):
            raise ValueError(
                f"boundary {face!r} must be 'outflow', 'wall', None or a callable"
            )
    return {face: value for face, value in bcs.items() if value not in (None, "outflow")}
# --- end m16803 boundary conditions ---


def _rhs(coeff, basis, derivative, weights, signs, dx, dy, periodic, geom=None,
         stage_time=0.0):
    """DG residual of the 2-D Euler equations on a Cartesian mesh."""
    degree = coeff.shape[2] - 1
    modes = torch.arange(degree + 1, device=coeff.device, dtype=coeff.dtype)
    outer = torch.outer(2.0 * modes + 1.0, 2.0 * modes + 1.0)

    at_quad = torch.einsum("xyklv,ak,bl->xyabv", coeff, basis, basis)
    volume_x = torch.einsum(
        "xyabv,a,b,ak,bl->xyklv", _physical_flux(at_quad, 0),
        weights, weights, derivative, basis,
    )
    volume_y = torch.einsum(
        "xyabv,a,b,ak,bl->xyklv", _physical_flux(at_quad, 1),
        weights, weights, basis, derivative,
    )

    right_x, left_x = _axis_fluxes(
        coeff, basis, signs, weights, periodic[0], 0, geom, stage_time
    )
    right_y, left_y = _axis_fluxes(
        coeff, basis, signs, weights, periodic[1], 1, geom, stage_time
    )

    face_x = (
        right_x.unsqueeze(2)
        - left_x.unsqueeze(2) * signs.view(1, 1, -1, 1, 1)
        - volume_x
    )
    face_y = (
        right_y.unsqueeze(3)
        - left_y.unsqueeze(3) * signs.view(1, 1, 1, -1, 1)
        - volume_y
    )
    scale = (outer / (2.0 * dx)).view(1, 1, *outer.shape, 1), (
        outer / (2.0 * dy)
    ).view(1, 1, *outer.shape, 1)
    return -(scale[0] * face_x) - (scale[1] * face_y)


# ---------------------------------------------------------------------------
# Shock limiting and positivity
# ---------------------------------------------------------------------------

def _shift(averages: torch.Tensor, axis: int, step: int, periodic: bool):
    """Neighbour averages along ``axis``.

    Periodic axes wrap (``torch.roll``); outflow axes clamp to the edge cell.
    Wrapping on a non-periodic axis would feed the opposite end of the mesh
    into the end cells, so the flag must be threaded through from the solver.
    Clamping alone is not enough: a clamped half stencil has a zero increment
    on one side, which makes any monotone profile look like an extremum, so
    ``_troubled_1d`` also drops the end cells of a non-periodic axis from the
    troubled set.
    """
    if periodic:
        return torch.roll(averages, shifts=-step, dims=axis)
    n = averages.shape[axis]
    index = (torch.arange(n, device=averages.device) + step).clamp(0, n - 1)
    return averages.index_select(axis, index)


def _troubled_1d(
    averages: torch.Tensor,
    axis: int,
    spacing: float,
    tvb_constant: float,
    periodic: bool = True,
):
    """Cockburn-Shu troubled-cell test along one axis on cell averages.

    The measure compared against the tolerance ``tvb_constant * spacing**2`` is
    the deviation ``|d_left - minmod(d_left, d_right)|``, which is O(dx**2 u'')
    for smooth data -- the same scale-consistent criterion the 1-D limiter has
    used since mission m2500 (``tensordg/tvb.py``).  Until mission m2549 the 2-D
    copy of this test compared the raw one-sided differences (O(dx u')) against
    ``M dx**2``, so every smooth extremum was flagged as soon as ``dx > 1/M``,
    more of them the finer the mesh, and ``tvb_constant`` was inert; the resulting
    order loss is documented in docs/limiter_2d_verification.md.

    Neighbour averages wrap around on periodic axes; on non-periodic axes the
    edge rows are never flagged (no full stencil), mirroring the 1-D limiter.
    """
    centres = averages
    plus = _shift(averages, axis, 1, periodic)
    minus = _shift(averages, axis, -1, periodic)
    delta_left = centres - minus
    delta_right = plus - centres
    troubled, minmod = troubled_from_differences(
        delta_left, delta_right, spacing, tvb_constant
    )
    if not periodic and averages.shape[axis] > 2:
        # No full stencil at an outflow face: leave the end cells unlimited
        # rather than deciding them from a replicated (zero-increment) ghost.
        cutter = [slice(None)] * troubled.dim()
        cutter[axis] = 0
        troubled[tuple(cutter)] = False
        cutter[axis] = -1
        troubled[tuple(cutter)] = False
    return troubled, minmod


def apply_tvb_limiter_2d(coeff, dx, dy, tvb_constant=1.0, periodic=(True, True)):
    """Directional minmod/TVB limiting of a tensor-product modal expansion.

    Troubled cells (cell averages whose one-sided differences deviate from the
    minmod slope by more than ``tvb_constant * spacing**2``) are reduced to a
    limited linear reconstruction along the offending direction; the higher
    modes of that direction are removed.  The cell average is never modified, so
    conservation is preserved exactly.

    Mission m2549 fixes two defects of this routine:

    * the troubled-cell test was not scale consistent (see ``_troubled_1d``);
    * for ``degree >= 2`` the mask that removes the higher modes was broadcast
      against the wrong axes, which raised ``RuntimeError`` for every degree >= 2
      case that actually had a troubled cell.  The four-quadrant verification of
      mission m2400 only ever ran the limiter at ``degree = 1``, so the broken
      branch was never exercised.

    ``periodic`` selects wrap-around (periodic axis) or edge replication with
    unflagged boundary rows (outflow axis) for the neighbour stencil.
    """
    if coeff.shape[2] <= 1 and coeff.shape[3] <= 1:
        return coeff
    averages = coeff[..., 0, 0, :]
    limited = coeff.clone()
    for axis, spacing in ((0, dx), (1, dy)):
        troubled, slope = _troubled_1d(
            averages, axis, spacing, tvb_constant, bool(periodic[axis])
        )
        if not troubled.any():
            continue
        mask = troubled.unsqueeze(-1)  # broadcast over the 4 components
        mask_modes = troubled.view(*troubled.shape, 1, 1, 1)  # (nx, ny, 1, 1, 1)
        if axis == 0:
            new_slope = 0.5 * slope
            limited[..., 1, 0, :] = torch.where(
                mask, new_slope, limited[..., 1, 0, :]
            )
            if coeff.shape[2] > 2:
                limited[..., 2:, :, :] = torch.where(
                    mask_modes, torch.zeros_like(limited[..., 2:, :, :]),
                    limited[..., 2:, :, :],
                )
        else:
            new_slope = 0.5 * slope
            limited[..., 0, 1, :] = torch.where(
                mask, new_slope, limited[..., 0, 1, :]
            )
            if coeff.shape[3] > 2:
                limited[..., :, 2:, :] = torch.where(
                    mask_modes, torch.zeros_like(limited[..., :, 2:, :]),
                    limited[..., :, 2:, :],
                )
    return limited


def _weno_slope_1d(averages, axis, spacing, periodic, epsilon=1e-6):
    """WENO3 non-linear slope (physical units) along ``axis``.

    Mirrors :func:`tensordg.limiters.apply_weno_limiter` line for line so that
    the two-dimensional limiter reduces exactly to the one-dimensional one on
    y-invariant data (see ``tests/test_limiter_2d.py``).
    """
    u_i = averages
    u_im1 = _shift(averages, axis, -1, periodic)
    u_ip1 = _shift(averages, axis, 1, periodic)
    slope0 = (u_i - u_im1) / spacing
    slope1 = (u_ip1 - u_i) / spacing
    slope2 = (u_ip1 - u_im1) / (2.0 * spacing)
    beta0 = spacing * slope0**2
    beta1 = spacing * slope1**2
    beta2 = (1.0 / 3.0) * (u_im1 - 2.0 * u_i + u_ip1) ** 2 / spacing + spacing * slope2**2
    beta_smooth = [beta.amax(dim=-1) for beta in (beta0, beta1, beta2)]
    weights = _weno_weights(*beta_smooth, epsilon=epsilon)
    return (
        weights[0][..., None] * slope0
        + weights[1][..., None] * slope1
        + weights[2][..., None] * slope2
    )


def apply_weno_limiter_2d(
    coeff, dx, dy, tvb_constant=1.0, periodic=(True, True), epsilon=1e-6
):
    """Dimension-by-dimension WENO limiting of a tensor-product DG expansion.

    For every axis, cells flagged by the same Cockburn-Shu indicator the TVB
    limiter uses are rebuilt from a WENO3 combination of the one-sided and
    centred slopes of the cell averages (Qiu & Shu, 2005, applied one
    dimension at a time).  Only modes of the offending axis are touched: mode 1
    receives the WENO slope and modes >= 2 are zeroed, while the modal
    content of the other axis is kept.  The cell average (mode 0 of both axes)
    is never modified, so conservation is preserved exactly.

    Mission m14777 fixed a slice-write defect: both axis passes assigned the
    whole ``[..., 1, :, :]`` / ``[..., :, 1, :]`` slice, so for ``degree >= 2``
    every mixed mode (1, j) (resp. (i, 1)) was overwritten with the
    one-dimensional slope, i.e. a spurious tensor-product term in every limited
    cell of degree >= 2.  Mission m14777 measured that repairing it leaves the
    vortex error norms bit-identical, so the limiter-on plateau in
    ``docs/anisotropic_2d_verification.md`` has a *different* source: at the
    default ``M`` the indicator legitimately flags resolved smooth cells and a
    limited (linear) reconstruction is only second order -- see
    ``docs/weno_limiter_smooth_flow_policy.md``.  Only mode ``(1, 0)`` / ``(0, 1)``
    is now touched, matching :func:`apply_tvb_limiter_2d`.
    """
    if coeff.shape[2] <= 1 and coeff.shape[3] <= 1:
        return coeff
    averages = coeff[..., 0, 0, :]
    limited = coeff
    for axis, spacing in ((0, dx), (1, dy)):
        if coeff.shape[2 + axis] <= 1:
            continue
        troubled, _ = _troubled_1d(
            averages, axis, spacing, tvb_constant, bool(periodic[axis])
        )
        if not troubled.any():
            continue
        slope = _weno_slope_1d(
            averages, axis, spacing, bool(periodic[axis]), epsilon
        )
        if limited is coeff:
            limited = coeff.clone()
        mask = troubled.unsqueeze(-1)  # (nx, ny, 1) broadcast over the 4 components
        mask_modes = troubled.view(*troubled.shape, 1, 1, 1)
        if axis == 0:
            # Only the *pure* mode-1 of the offending axis is replaced.  Writing
            # the whole ``[..., 1, :, :]`` slice (m14777 defect) stamped the
            # one-dimensional slope into every mixed mode (1, j >= 1) and
            # destroyed the tensor-product content of the other axis.
            limited[..., 1, 0, :] = torch.where(
                mask, (0.5 * spacing) * slope, limited[..., 1, 0, :]
            )
            if coeff.shape[2] > 2:
                limited[..., 2:, :, :] = torch.where(
                    mask_modes,
                    torch.zeros_like(limited[..., 2:, :, :]),
                    limited[..., 2:, :, :],
                )
        else:
            limited[..., 0, 1, :] = torch.where(
                mask, (0.5 * spacing) * slope, limited[..., 0, 1, :]
            )
            if coeff.shape[3] > 2:
                limited[..., :, 2:, :] = torch.where(
                    mask_modes,
                    torch.zeros_like(limited[..., :, 2:, :]),
                    limited[..., :, 2:, :],
                )
    return limited


def _admissible(state: torch.Tensor) -> torch.Tensor:
    """Element-wise mask: density and pressure positive and state finite."""
    rho = state[..., 0]
    pressure = _pressure(state)
    return (rho > 0.0) & (pressure > 0.0) & torch.isfinite(state).all(-1)


def _all_over_sample_points(mask: torch.Tensor) -> torch.Tensor:
    """Reduce a point mask of shape ``(nx, ny, ...)`` to one flag per cell."""
    return mask.reshape(*mask.shape[:2], -1).all(dim=-1)


def _positivity_filter(coeff, basis, signs):
    """Collapse to the cell average wherever the solution is non-positive.

    Admissibility is tested on the cell average, on every volume quadrature
    node and on both traces in each direction, because the interface flux is
    evaluated from the trace polynomials and would otherwise receive
    non-positive states.
    """
    averages = coeff[..., 0, 0, :]
    bad = ~_all_over_sample_points(_admissible(averages))
    at_quad = torch.einsum("xyklv,ak,bl->xyabv", coeff, basis, basis)
    bad = bad | ~_all_over_sample_points(_admissible(at_quad))
    for axis in (0, 1):
        trace_right, trace_left = _traces(coeff, basis, signs, axis)
        bad = bad | ~_all_over_sample_points(_admissible(trace_right))
        bad = bad | ~_all_over_sample_points(_admissible(trace_left))
    if bad.any():
        average_only = torch.zeros_like(coeff)
        average_only[..., 0, 0, :] = averages
        mask = bad.view(*bad.shape, *([1] * (coeff.dim() - bad.dim())))
        coeff = torch.where(mask, average_only, coeff)
    return coeff


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _initial_averages(initial, x, y, dx, dy, device, dtype, quadrature=4):
    nodes_np, weights_np = np.polynomial.legendre.leggauss(quadrature)
    nodes = torch.as_tensor(nodes_np, device=device, dtype=dtype)
    weights = torch.as_tensor(weights_np, device=device, dtype=dtype)
    xq = x[:, None, None, None] + 0.5 * dx * nodes[None, None, :, None]
    yq = y[None, :, None, None] + 0.5 * dy * nodes[None, None, None, :]
    state = _conservative(*initial(xq, yq))
    return 0.25 * torch.sum(
        state * weights[None, None, :, None, None] * weights[None, None, None, :, None],
        dim=(2, 3),
    )


def _initial_coefficients(initial, x, y, dx, dy, nodes, weights, basis, device, dtype):
    """L2 projection of ``initial(x, y) -> (rho, u, v, p)`` onto the modal basis.

    ``c_kl = (2k+1)(2l+1)/4 * int int u(xi, eta) P_k(xi) P_l(eta) dxi deta``,
    evaluated with the same Gauss quadrature as the residual.  Projecting onto
    *all* modes (instead of storing cell averages only) is what preserves the
    design order of the scheme for smooth initial data.
    """
    xq = x[:, None, None, None] + 0.5 * dx * nodes[None, None, :, None]
    yq = y[None, :, None, None] + 0.5 * dy * nodes[None, None, None, :]
    state = _conservative(*initial(xq, yq))
    projected = torch.einsum(
        "xyabv,a,b,ak,bl->xyklv", state, weights, weights, basis, basis
    )
    degree = basis.shape[1] - 1
    modes = torch.arange(degree + 1, device=device, dtype=dtype)
    return 0.25 * torch.outer(2.0 * modes + 1.0, 2.0 * modes + 1.0).view(
        1, 1, degree + 1, degree + 1, 1
    ) * projected


def solve_euler_dg_2d(
    initial,
    ncells=32,
    degree=2,
    domain=((0.0, 1.0), (0.0, 1.0)),
    final_time=0.2,
    cfl=0.2,
    dt=None,
    limiter=None,
    tvb_constant=1.0,
    limiter_stage="every",
    periodic=(True, True),
    quadrature=None,
    bcs=None,
    initial_projection="l2",
    device="cpu",
    dtype=torch.float64,
):
    """Advance a 2-D Euler initial condition with modal DG / HLLC / SSP-RK3.

    Parameters
    ----------
    initial : callable
        ``initial(x, y) -> (rho, u, v, p)`` evaluated on tensors of any shape.
    ncells : int or (int, int)
        Cell counts in ``x`` and ``y``.
    degree : int
        Polynomial degree of the tensor-product basis (0, 1, 2, 3 supported).
    dt : float or None
        Optional *fixed* time step.  ``None`` (default) keeps the adaptive
        ``cfl`` step, which is the production path.  A float marches with a
        constant step of at most that value (the last step is trimmed to land
        exactly on ``final_time``); it is a verification knob for temporal
        refinement studies -- the caller is responsible for keeping the step
        inside the explicit stability limit.
    limiter : str or None
        ``None`` (positivity fallback only), ``"tvb"`` or ``"weno"``.
    tvb_constant : float
        TVB constant M; M=0 is strict TVD, larger M allows smooth extrema.
    limiter_stage : {"every", "final"}
        When the limiter is applied inside the SSP-RK3 sweep.  ``"every"``
        (default, historic behaviour) limits after every sub-step, which
        re-projects smooth cells that keep being flagged and therefore
        destroys high-order accuracy there (see
        ``docs/limiter_verification.md``).  ``"final"`` limits only the
        updated state at the end of the time step.
    periodic : (bool, bool)
        Periodicity in ``x`` and ``y``; non-periodic faces are outflow unless
        ``bcs`` overrides them.
    bcs : dict or None
        Boundary conditions for the non-periodic faces, keyed by ``"left"`` /
        ``"right"`` / ``"bottom"`` / ``"top"``.  Each value is ``"outflow"``
        (default), ``"wall"`` (impermeable inviscid slip wall, exact wall flux
        ``(0, p, 0, 0)``) or ``callable(X, Y, t)`` returning a conservative
        ghost state, or ``(state, wall_mask)`` where ``True`` switches that
        point to the wall flux.  ``X``/``Y`` are the physical coordinates of
        the face quadrature points, shape ``(n_face, quadrature)``; ``t`` is the
        RK stage time.  Entries of a periodic axis are ignored.
    initial_projection : str
        ``"l2"`` (default, full modal L2 projection -- preserves design order)
        or ``"cell-average"`` (P0 only, the legacy m2143 behaviour).
    """
    nx, ny = (ncells, ncells) if isinstance(ncells, int) else ncells
    if degree not in (0, 1, 2, 3):
        raise ValueError("degree must be 0, 1, 2 or 3")
    if limiter_stage not in ("every", "final"):
        raise ValueError("limiter_stage must be 'every' or 'final'")
    if nx < 1 or ny < 1 or final_time < 0.0 or not 0.0 < cfl <= 0.5:
        raise ValueError("invalid cells, final_time or CFL")
    prescribed_dt = None if dt is None else float(dt)
    if prescribed_dt is not None and not prescribed_dt > 0.0:
        raise ValueError("dt must be positive when prescribed")
    (xmin, xmax), (ymin, ymax) = domain
    if not xmax > xmin or not ymax > ymin:
        raise ValueError("each domain interval must be increasing")
    periodic_flags_pre = (bool(periodic[0]), bool(periodic[1]))
    periodic_flags = periodic_flags_pre
    dev = torch.device(device)
    order = quadrature if quadrature is not None else max(degree + 2, 3)
    nodes, weights, basis, derivative, signs = legendre_operators(
        degree, order, dev, dtype
    )
    dx = (xmax - xmin) / nx
    dy = (ymax - ymin) / ny
    x = xmin + (torch.arange(nx, device=dev, dtype=dtype) + 0.5) * dx
    y = ymin + (torch.arange(ny, device=dev, dtype=dtype) + 0.5) * dy

    face_bcs = normalise_bcs(bcs)
    for face, axis in (("left", 0), ("right", 0), ("bottom", 1), ("top", 1)):
        if face in face_bcs and periodic_flags_pre[axis]:
            face_bcs.pop(face)  # periodic axis wins; documented above
    geom = {
        "nodes": nodes,
        "x": x,
        "y": y,
        "dx": dx,
        "dy": dy,
        "xmin": xmin,
        "xmax": xmax,
        "ymin": ymin,
        "ymax": ymax,
        "nx": nx,
        "ny": ny,
        "bcs": face_bcs,
    }

    if initial_projection == "l2":
        coeff = _initial_coefficients(
            initial, x, y, dx, dy, nodes, weights, basis, dev, dtype
        )
    elif initial_projection == "cell-average":
        averages = _initial_averages(initial, x, y, dx, dy, dev, dtype)
        coeff = torch.zeros((nx, ny, degree + 1, degree + 1, 4), device=dev, dtype=dtype)
        coeff[..., 0, 0, :] = averages
    else:
        raise ValueError("initial_projection must be 'l2' or 'cell-average'")

    limit_each_stage = limiter_stage == "every"

    def _apply_limiter(state):
        if limiter == "tvb":
            state = apply_tvb_limiter_2d(
                state, dx, dy, tvb_constant, periodic_flags
            )
        elif limiter == "weno":
            state = apply_weno_limiter_2d(
                state, dx, dy, tvb_constant, periodic_flags
            )
        elif limiter is not None:
            raise ValueError("limiter must be None, 'tvb' or 'weno'")
        return state

    def impose(state):
        state = _apply_limiter(state)
        return _positivity_filter(state, basis, signs)

    time, steps, dt = 0.0, 0, 0.0
    while time < final_time:
        at_quad = torch.einsum("xyklv,ak,bl->xyabv", coeff, basis, basis)
        _, velocity_x, velocity_y, pressure = _primitive(at_quad)
        sound = torch.sqrt(GAMMA * pressure.clamp_min(1e-12) / at_quad[..., 0].clamp_min(1e-12))
        speed_x = float(torch.max(velocity_x.abs() + sound).item())
        speed_y = float(torch.max(velocity_y.abs() + sound).item())
        rate = (speed_x / dx + speed_y / dy) * (2 * degree + 1)
        if prescribed_dt is None:
            dt = min(cfl / max(rate, 1e-12), final_time - time)
        else:
            dt = min(prescribed_dt, final_time - time)
        k1 = _rhs(coeff, basis, derivative, weights, signs, dx, dy, periodic, geom, time)
        u1 = coeff + dt * k1
        if limit_each_stage:
            u1 = impose(u1)
        else:
            u1 = _positivity_filter(u1, basis, signs)
        k2 = _rhs(
            u1, basis, derivative, weights, signs, dx, dy, periodic, geom, time + dt
        )
        u2 = 0.75 * coeff + 0.25 * (u1 + dt * k2)
        if limit_each_stage:
            u2 = impose(u2)
        else:
            u2 = _positivity_filter(u2, basis, signs)
        k3 = _rhs(
            u2, basis, derivative, weights, signs, dx, dy, periodic, geom, time + 0.5 * dt
        )
        coeff = impose(coeff / 3.0 + (2.0 / 3.0) * (u2 + dt * k3))
        if not torch.isfinite(coeff).all():
            raise FloatingPointError("non-finite 2-D DG state; reduce the CFL number")
        rho_now = coeff[..., 0, 0, 0]
        pressure_now = _pressure(coeff[..., 0, 0, :])
        if (rho_now <= 0.0).any() or (pressure_now <= 0.0).any():
            raise FloatingPointError(
                "nonphysical cell average in 2-D DG; reduce the CFL number"
            )
        time += dt
        steps += 1

    metadata = {
        "ncells": (nx, ny),
        "degree": degree,
        "final_time": final_time,
        "steps": steps,
        "dt": dt,
        "cfl": cfl,
        "dt_mode": "cfl" if prescribed_dt is None else "fixed",
        "prescribed_dt": prescribed_dt,
        "domain": domain,
        "periodic": tuple(periodic),
        "bcs": {
            face: (value if isinstance(value, str) else getattr(value, "__name__", "callable"))
            for face, value in face_bcs.items()
        },
        "flux": "hllc",
        "limiter": limiter if limiter else "local positivity fallback",
        "limiter_stage": limiter_stage,
        "quadrature": order,
        "initial_projection": initial_projection,
        "device": str(dev),
        "dtype": str(dtype).replace("torch.", ""),
        "dofs": nx * ny * (degree + 1) ** 2 * 4,
    }
    return x, y, coeff, metadata


def cell_fields(coeff: torch.Tensor):
    """Return cell-average primitive fields (density, velocities, pressure)."""
    averages = coeff[..., 0, 0, :]
    rho, velocity_x, velocity_y, pressure = _primitive(averages)
    return {
        "density": rho,
        "velocity_x": velocity_x,
        "velocity_y": velocity_y,
        "pressure": pressure,
        "conservative": averages,
    }


# ---------------------------------------------------------------------------
# Named benchmark set-ups
# ---------------------------------------------------------------------------

def solve_euler_dg_2d_vortex(
    ncells=32,
    degree=2,
    final_time=1.0,
    cfl=0.2,
    dt=None,
    strength=5.0,
    background_velocity=(1.0, 1.0),
    limiter=None,
    tvb_constant=1.0,
    limiter_stage="every",
    domain=((-5.0, 5.0), (-5.0, 5.0)),
    initial_projection="l2",
    device="cpu",
    dtype=torch.float64,
):
    """Periodic smooth isentropic vortex on a high-order 2-D DG mesh.

    ``limiter_stage`` is forwarded to :func:`solve_euler_dg_2d` (see its
    docstring): ``"every"`` limits at every SSP-RK3 sub-step (historic
    behaviour), ``"final"`` limits only the end-of-step state.
    """

    def initial(x, y):
        return vortex_primitive(x, y, 0.0, strength, background_velocity)

    return solve_euler_dg_2d(
        initial,
        ncells=ncells,
        degree=degree,
        domain=domain,
        final_time=final_time,
        cfl=cfl,
        dt=dt,
        limiter=limiter,
        tvb_constant=tvb_constant,
        limiter_stage=limiter_stage,
        periodic=(True, True),
        initial_projection=initial_projection,
        device=device,
        dtype=dtype,
    )


def solve_euler_dg_2d_sod(
    ncells=200,
    degree=2,
    final_time=0.2,
    cfl=0.2,
    limiter=None,
    tvb_constant=1.0,
    diaphragm=0.5,
    left_state=(1.0, 0.0, 1.0),
    right_state=(0.125, 0.0, 0.1),
    domain=((0.0, 1.0), (0.0, 0.25)),
    initial_projection="l2",
    device="cpu",
    dtype=torch.float64,
):
    """Sod tube extruded to 2-D (one cell thick, y-invariant data).

    Because the initial data do not depend on ``y`` and the flux difference
    across the two ``y`` faces cancels exactly, the discrete solution must
    reproduce the 1-D DG solution cell by cell.  This makes the set-up a
    cross-code verification of the 2-D residual assembly.
    """

    def initial(x, y):
        shape = torch.broadcast_shapes(x.shape, y.shape)
        mask = torch.broadcast_to(x < diaphragm, shape)
        rho = torch.where(mask, torch.full(shape, left_state[0], dtype=x.dtype),
                          torch.full(shape, right_state[0], dtype=x.dtype))
        velocity_x = torch.where(mask, torch.full(shape, left_state[1], dtype=x.dtype),
                                 torch.full(shape, right_state[1], dtype=x.dtype))
        pressure = torch.where(mask, torch.full(shape, left_state[2], dtype=x.dtype),
                               torch.full(shape, right_state[2], dtype=x.dtype))
        return rho, velocity_x, torch.zeros_like(rho), pressure

    return solve_euler_dg_2d(
        initial,
        ncells=(ncells, 1),
        degree=degree,
        domain=domain,
        final_time=final_time,
        cfl=cfl,
        limiter=limiter,
        tvb_constant=tvb_constant,
        periodic=(False, True),
        initial_projection=initial_projection,
        device=device,
        dtype=dtype,
    )


# Four-quadrant Riemann problem, Lax & Liu (1998) configuration 3
# (interaction of two shocks).  Quadrant order: (lower-left, lower-right,
# upper-left, upper-right) with states (rho, u, v, p).
LAX_LIU_CONFIG_3 = {
    "lower-left": (0.138, 1.206, 1.206, 0.3),
    "lower-right": (0.5323, 0.0, 1.206, 0.3),
    "upper-left": (0.5323, 1.206, 0.0, 0.3),
    "upper-right": (1.5, 0.0, 0.0, 1.5),
}


def solve_euler_dg_2d_riemann(
    ncells=100,
    degree=2,
    final_time=0.25,
    cfl=0.2,
    limiter="tvb",
    tvb_constant=1.0,
    states=None,
    initial_projection="l2",
    device="cpu",
    dtype=torch.float64,
):
    """Four-quadrant 2-D Riemann problem (Lax & Liu 1998, configuration 3)."""
    quadrants = LAX_LIU_CONFIG_3 if states is None else states

    def initial(x, y):
        shape = torch.broadcast_shapes(x.shape, y.shape)
        lower = torch.broadcast_to(y < 0.5, shape)
        left = torch.broadcast_to(x < 0.5, shape)
        chosen = (
            ("lower-left", lower & left),
            ("lower-right", lower & ~left),
            ("upper-left", ~lower & left),
            ("upper-right", ~lower & ~left),
        )
        rho = torch.zeros(shape, dtype=x.dtype)
        velocity_x = torch.zeros(shape, dtype=x.dtype)
        velocity_y = torch.zeros(shape, dtype=x.dtype)
        pressure = torch.zeros(shape, dtype=x.dtype)
        for name, mask in chosen:
            rho = torch.where(mask, torch.full(shape, quadrants[name][0], dtype=x.dtype), rho)
            velocity_x = torch.where(mask, torch.full(shape, quadrants[name][1], dtype=x.dtype), velocity_x)
            velocity_y = torch.where(mask, torch.full(shape, quadrants[name][2], dtype=x.dtype), velocity_y)
            pressure = torch.where(mask, torch.full(shape, quadrants[name][3], dtype=x.dtype), pressure)
        return rho, velocity_x, velocity_y, pressure

    return solve_euler_dg_2d(
        initial,
        ncells=ncells,
        degree=degree,
        domain=((0.0, 1.0), (0.0, 1.0)),
        final_time=final_time,
        cfl=cfl,
        limiter=limiter,
        tvb_constant=tvb_constant,
        periodic=(False, False),
        initial_projection=initial_projection,
        device=device,
        dtype=dtype,
    )