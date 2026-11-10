"""One-dimensional shallow water equations with DG(P1/P2).

Solves the 1-D SWE system with bed slope source term:

    ∂h/∂t + ∂(hu)/∂x = 0
    ∂(hu)/∂t + ∂(hu² + ½gh²)/∂x = −g h ∂z_b/∂x

Conservative variables U = [h, hu], physical flux F = [hu, hu² + ½gh²].

Features
--------
* Modal Legendre DG of degree 1 or 2 (P1/P2).
* HLL or Lax-Friedrichs numerical flux at interfaces.
* Well-balanced treatment: source term projected consistently with flux.
* Minmod/TVB limiter for shock capturing.
* SSP-RK2 / SSP-RK3 time integration.
* Positivity-preserving filter (h ≥ 0, Zhang-Shu scaling of the higher modes
  towards the cell average, so admissible cells -- including a dry-bank
  shoreline -- pass through unchanged).

Analytical benchmarks
---------------------
* Lake-at-rest: η = h + z_b = const, u = 0 → zero drift to machine precision.
* Ritter dam break (flat bed, dry downstream): exact self-similar solution.
* Thacker (1981) parabolic basin oscillation: periodic analytical solution.

References
----------
* Audusse et al. (2004), "A fast and stable well-balanced scheme..."
* Ritter (1892), "Über die Fortpflanzung von Wasserwellen."
* Thacker (1981), "Some exact solutions of the nonlinear shallow-water equations."
"""
from __future__ import annotations

import math
import warnings
from typing import Callable, Optional

import numpy as np
import torch

from .limiters import apply_limiter

GRAVITY = 9.81  # m/s²

#: Depth (m) below which a cell or interface state is treated as *dry*.
#: The velocity of a dry state is defined as zero, which keeps ``hu / h``
#: finite when a higher-order reconstruction dips below the still-water level
#: (Gibbs undershoot at a wetting front) and keeps the CFL estimate meaningful.
DRY_DEPTH = 1e-6


def _primitive(state: torch.Tensor):
    """Dry-guarded primitives ``(h, u, c)`` for a conservative state ``[h, hu]``.

    ``h`` is clamped at zero, ``u = hu / max(h, DRY_DEPTH)`` for wet states and
    ``u = 0`` for dry states (``h <= DRY_DEPTH``), and ``c = sqrt(g h)``.
    The split keeps every quantity finite when a modal reconstruction dips
    below the still-water level, instead of evaluating ``hu / 1e-12``.
    """
    h = state[..., 0].clamp_min(0.0)
    u = torch.where(h > DRY_DEPTH, state[..., 1] / h.clamp_min(DRY_DEPTH),
                    torch.zeros_like(h))
    return h, u, torch.sqrt(GRAVITY * h)


def physical_flux(u: torch.Tensor) -> torch.Tensor:
    """Physical flux F(U) for U = [h, hu].

    The momentum flux is evaluated as ``h u^2 + g h^2 / 2`` with the dry-guarded
    velocity of :func:`_primitive`, so that a vanishing depth carries a
    vanishing flux.  The algebraically equivalent form ``hu^2 / h + g h^2 / 2``
    is *not* used: at a wetting front ``hu`` need not vanish with ``h`` and the
    quotient then amplifies round-off into a catastrophic flux.
    """
    h, vel, _ = _primitive(u)
    return torch.stack((h * vel, h * vel * vel + 0.5 * GRAVITY * h ** 2), dim=-1)


def sound_speed(u: torch.Tensor) -> torch.Tensor:
    """Linear wave speed c = sqrt(g h)."""
    return torch.sqrt(GRAVITY * u[..., 0].clamp_min(1e-12))


def hll_flux(left: torch.Tensor, right: torch.Tensor, flux_kind: str = "hll") -> torch.Tensor:
    """HLL numerical flux for the SWE system.

    Uses Davis wave-speed estimates. Falls back to Lax-Friedrichs if
    ``flux_kind == "lf"``.
    """
    if flux_kind == "lf":
        fl = physical_flux(left)
        fr = physical_flux(right)
        _, u_l, c_l = _primitive(left)
        _, u_r, c_r = _primitive(right)
        alpha = torch.maximum(torch.abs(u_l) + c_l, torch.abs(u_r) + c_r)
        return 0.5 * (fl + fr) - 0.5 * alpha[..., None] * (right - left)

    _, u_l, c_l = _primitive(left)
    _, u_r, c_r = _primitive(right)

    s_l = torch.minimum(u_l - c_l, u_r - c_r)  # [ncells]
    s_r = torch.maximum(u_l + c_l, u_r + c_r)  # [ncells]

    fl = physical_flux(left)   # [ncells, 2]
    fr = physical_flux(right)  # [ncells, 2]

    denom = s_r - s_l  # [ncells]
    safe_denom = torch.where(denom.abs() < 1e-30, torch.full_like(denom, 1e-30), denom)

    # HLL flux: (s_r * F_L - s_l * F_R + s_l * s_r * (U_R - U_L)) / (s_r - s_l)
    # Need [..., None] to broadcast [ncells] with [ncells, 2]
    flux = (s_r[..., None] * fl - s_l[..., None] * fr
            + s_l[..., None] * s_r[..., None] * (right - left)) / safe_denom[..., None]

    flux = torch.where((s_l >= 0.0)[..., None], fl, flux)
    flux = torch.where((s_r <= 0.0)[..., None], fr, flux)
    return flux


def _operators(degree: int, device: torch.device, dtype: torch.dtype):
    """Build Legendre basis, derivatives, and boundary values."""
    nq = max(2 * degree + 3, 4)
    nodes_np, weights_np = np.polynomial.legendre.leggauss(nq)
    nodes = torch.as_tensor(nodes_np, device=device, dtype=dtype)
    weights = torch.as_tensor(weights_np, device=device, dtype=dtype)

    values, derivatives = [], []
    for k in range(degree + 1):
        poly = np.zeros(k + 1)
        poly[-1] = 1.0
        deriv = np.polynomial.legendre.legder(poly)
        values.append(torch.as_tensor(
            np.polynomial.legendre.legval(nodes_np, poly), device=device, dtype=dtype,
        ))
        deriv_vals = (np.polynomial.legendre.legval(nodes_np, deriv)
                      if len(deriv) else np.zeros_like(nodes_np))
        derivatives.append(torch.as_tensor(deriv_vals, device=device, dtype=dtype))

    phi = torch.stack(values, dim=1)
    dphi = torch.stack(derivatives, dim=1)
    phi_left = torch.tensor([(-1.0) ** k for k in range(degree + 1)], device=device, dtype=dtype)
    phi_right = torch.ones(degree + 1, device=device, dtype=dtype)
    return nodes, weights, phi, dphi, phi_left, phi_right


def _lgl_transform(degree: int, device: torch.device, dtype: torch.dtype):
    """Nodal -> modal transform for the degree-``p`` Gauss-Lobatto-Legendre nodes.

    Returns ``(V^{-1}, nodes)`` with ``V[j, k] = P_k(xi_j)`` for the ``p + 1``
    LGL nodes ``xi_j`` (which include both endpoints ``-1`` and ``+1``).  For a
    nodal vector ``v_j`` of values at those nodes the modal coefficients are
    ``c = V^{-1} v``; the polynomial reproduces ``v`` *exactly* at the nodes,
    in particular at the two cell faces.

    Only ``p = 1`` (nodes ``-1, +1``) and ``p = 2`` (nodes ``-1, 0, +1``) are
    needed by this solver.
    """
    nodes_np = (np.array([-1.0, 1.0]) if degree == 1
                else np.array([-1.0, 0.0, 1.0]))
    vander = np.stack(
        [np.polynomial.legendre.legval(nodes_np, e) for e in np.eye(degree + 1)],
        axis=1)
    vander_t = torch.as_tensor(vander, device=device, dtype=dtype)
    return torch.linalg.inv(vander_t), torch.as_tensor(nodes_np, device=device, dtype=dtype)


def _hydrostatic_interface_flux(coeff: torch.Tensor, phi_left: torch.Tensor,
                                phi_right: torch.Tensor,
                                bed_modal: torch.Tensor,
                                flux_kind: str = "hll",
                                periodic: bool = False) -> tuple:
    """Audusse et al. (2004) hydrostatic reconstruction at every interface.

    For each face ``i+1/2`` the two one-sided traces are reconstructed onto the
    *higher* of the two face bed elevations

        z_b* = max(z_b,iR, z_b,i+1,L),   h*_m = max(0, h_m + z_b,m - z_b*),

    the numerical flux is evaluated with the reconstructed states, and the
    hydrostatic pressure correction ``g/2 (h_m^2 - h*_m^2)`` is added back to
    the momentum component *of that side*.  The result is the pair of one-sided
    fluxes ``(F_i,R, F_i+1,L)``; the two differ only through the correction
    term, and at rest each side sees the pressure flux ``g/2 h_m^2`` of its own
    trace.  That is what makes a lake at rest a discrete fixed point across a
    bed *jump*: the flux divergence then cancels the exact bed-slope source of
    each cell individually, exactly as on a continuous bed.

    For a continuous bed ``z_b,iR = z_b,i+1,L``, hence ``h* = h`` and the
    correction vanishes identically: the scheme reduces algebraically to the
    shared-face HLL flux of the default path.  On a wet/dry interface the
    reconstruction removes the spurious mass flux that a plain HLL flux would
    produce between a wet trace and a dry neighbour whose bed is above the
    still-water level.

    Domain ends use the zero-order extrapolation (transmissive) closure of the
    default path; the analytic-boundary-flux override is applied by the caller.

    Returns ``(flux_right_of_cell, flux_left_of_cell)``, each ``(ncells, 2)``.
    """
    right_trace = torch.einsum("cmv,m->cv", coeff, phi_right)
    left_trace = torch.einsum("cmv,m->cv", coeff, phi_left)
    bed_right = torch.einsum("cm,m->c", bed_modal, phi_right)
    bed_left = torch.einsum("cm,m->c", bed_modal, phi_left)

    # Right face of cell i paired with the left face of cell i+1 (or ghost).
    h_a, hu_a, b_a = right_trace[:, 0], right_trace[:, 1], bed_right
    h_b = torch.roll(left_trace[:, 0], shifts=-1, dims=0)
    hu_b = torch.roll(left_trace[:, 1], shifts=-1, dims=0)
    b_b = torch.roll(bed_left, shifts=-1, dims=0)
    if not periodic:
        h_b[-1], hu_b[-1], b_b[-1] = h_a[-1], hu_a[-1], b_a[-1]

    b_star = torch.maximum(b_a, b_b)
    h_a_pos = h_a.clamp_min(0.0)
    h_b_pos = h_b.clamp_min(0.0)
    h_a_star = (h_a_pos + b_a - b_star).clamp_min(0.0)
    h_b_star = (h_b_pos + b_b - b_star).clamp_min(0.0)
    u_a = torch.where(h_a_pos > DRY_DEPTH, hu_a / h_a_pos.clamp_min(DRY_DEPTH),
                      torch.zeros_like(h_a))
    u_b = torch.where(h_b_pos > DRY_DEPTH, hu_b / h_b_pos.clamp_min(DRY_DEPTH),
                      torch.zeros_like(h_b))

    state_a = torch.stack((h_a_star, h_a_star * u_a), dim=-1)
    state_b = torch.stack((h_b_star, h_b_star * u_b), dim=-1)
    flux = hll_flux(state_a, state_b, flux_kind)

    corr_a = 0.5 * GRAVITY * (h_a_pos ** 2 - h_a_star ** 2)
    corr_b = 0.5 * GRAVITY * (h_b_pos ** 2 - h_b_star ** 2)
    flux_right = torch.stack((flux[:, 0], flux[:, 1] + corr_a), dim=-1)
    flux_left_next = torch.stack((flux[:, 0], flux[:, 1] + corr_b), dim=-1)

    flux_left = torch.roll(flux_left_next, shifts=1, dims=0)
    if not periodic:
        flux_left[0] = physical_flux(left_trace[:1])[0]
        flux_right[-1] = physical_flux(right_trace[-1:])[0]
    return flux_right, flux_left


def _nodal_values(fn, node_x: torch.Tensor, name: str) -> torch.Tensor:
    """Sample ``fn`` on the cell-local nodes, or validate pre-sampled data.

    ``fn`` is normally a callable of ``x``.  A tensor of shape
    ``(ncells, n_nodes)`` is accepted as already-sampled *cell-local* nodal
    data; that is the only way to describe a bed (or an initial depth) that is
    discontinuous at a shared cell face, because a pointwise callable cannot
    return two different values at the same face coordinate.
    """
    if callable(fn):
        return fn(node_x)
    value = torch.as_tensor(fn, device=node_x.device, dtype=node_x.dtype)
    if value.shape != node_x.shape:
        raise ValueError(
            f"{name} must be a callable of x or a tensor of shape "
            f"{tuple(node_x.shape)} (cells x nodes), got {tuple(value.shape)}")
    return value


def _rhs(coeff: torch.Tensor, dx: float, weights: torch.Tensor,
         phi: torch.Tensor, dphi: torch.Tensor,
         phi_left: torch.Tensor, phi_right: torch.Tensor,
         zb_jump: torch.Tensor,
         flux_kind: str = "hll", periodic: bool = False,
         time: float = 0.0,
         bc_left_flux=None, bc_right_flux=None,
         zb_slope_quad: Optional[torch.Tensor] = None,
         bed_modal: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Semi-discrete DG residual for SWE with bed slope source term.

    The source term −g h ∂z_b/∂x is treated by modal projection.
    For piecewise-linear bed, ∂z_b/∂x = (z_b,R − z_b,L)/dx is constant per cell.
    The weak-form source for mode k in the momentum equation is:
        S_k = −g (∂z_b/∂x) · c_k^h     (the (2k+1)/dx weak-form prefactor
        cancels the 2/(2k+1) modal moment, so the source is weight-free)
    where c_k^h is the k-th modal coefficient of h.

    For lake-at-rest (h + z_b = const, hu = 0), the flux gradient and source
    cancel exactly at the discrete level, giving machine-precision well-balancedness.
    """
    at_quad = torch.einsum("cmv,qm->cqv", coeff, phi)
    flux_quad = physical_flux(at_quad)
    volume = torch.einsum("cqv,q,qk->ckv", flux_quad, weights, dphi)

    right_trace = torch.einsum("cmv,m->cv", coeff, phi_right)
    left_trace = torch.einsum("cmv,m->cv", coeff, phi_left)

    if bed_modal is None:
        right_neighbour = torch.roll(left_trace, shifts=-1, dims=0)
        if not periodic:
            right_neighbour[-1] = right_trace[-1]

        interface_flux = hll_flux(right_trace, right_neighbour, flux_kind)
        left_interface_flux = torch.roll(interface_flux, shifts=1, dims=0)
        if not periodic:
            left_interface_flux[0] = hll_flux(left_trace[:1], left_trace[:1],
                                              flux_kind)
    else:
        # Well-balanced branch: Audusse hydrostatic reconstruction.  The flux
        # is one-sided per cell, so the two traces no longer share one value.
        interface_flux, left_interface_flux = _hydrostatic_interface_flux(
            coeff, phi_left, phi_right, bed_modal, flux_kind, periodic)

    # Optional exact boundary data (verification against analytic solutions).
    if bc_left_flux is not None:
        left_interface_flux[0] = torch.as_tensor(
            bc_left_flux(time), dtype=coeff.dtype, device=coeff.device)
    if bc_right_flux is not None:
        interface_flux[-1] = torch.as_tensor(
            bc_right_flux(time), dtype=coeff.dtype, device=coeff.device)

    dbdx = zb_jump / dx  # [ncells]
    if bed_modal is not None and zb_slope_quad is None:
        # dz_b/dx of the per-cell bed *polynomial* at the same Gauss points as h
        zb_slope_quad = (2.0 / dx) * torch.einsum("cm,qm->cq", bed_modal, dphi)

    # Two source-term discretisations:
    #   * legacy (zb_slope_quad is None): cell-constant gradient
    #     (z_b,R - z_b,L)/dx.  Exact for a piecewise-linear bed, but for a
    #     curved bed it is only the *cell average* of dz_b/dx, so the weak
    #     force carries an O(dx * d2z_b/dx2) error that does not shrink with p.
    #   * exact (zb_slope_quad given): dz_b/dx sampled at the same Gauss points
    #     as h, so the source quadrature is exact for polynomial data:
    #         S_k = -(2k+1)/2 g sum_q w_q h(x_q) dz_b/dx(x_q) phi_k(xi_q).
    #     For a constant slope this reduces algebraically to -g dbdx c_k, i.e.
    #     the legacy branch is recovered exactly.
    mom_factor = torch.arange(1, 2 * coeff.shape[1], 2,
                              device=coeff.device, dtype=coeff.dtype)  # 2k+1

    out = torch.empty_like(coeff)
    for k in range(coeff.shape[1]):
        # Flux divergence term (both equations)
        rhs_k = ((2 * k + 1) / dx) * (
            volume[:, k]
            - interface_flux * phi_right[k]
            + left_interface_flux * phi_left[k]
        )
        # Source term: only in momentum equation (component 1)
        # Weak form of -g h dz_b/dx on mode k:
        #   ((2k+1)/dx) * int_cell S phi_k dx with S = -g dbdx * sum_m c_m^h phi_m
        #   = -g dbdx * c_k^h  ((2k+1)/2 and 2/(2k+1) cancel exactly)
        if zb_slope_quad is None:
            src_k = -GRAVITY * dbdx * coeff[:, k, 0]
        else:
            src_k = -0.5 * GRAVITY * mom_factor[k] * torch.einsum(
                "cq,q,q->c", at_quad[:, :, 0] * zb_slope_quad, weights, phi[:, k])
        rhs_k[:, 1] += src_k
        out[:, k] = rhs_k

    return out


def _positivity_filter(coeff: torch.Tensor, check_points: torch.Tensor,
                       h_min: float = 1e-12,
                       well_balanced: bool = False) -> torch.Tensor:
    """Keep the water depth non-negative at every quadrature/interface point.

    Two actions are available and they are *not* interchangeable:

    ``well_balanced=True`` -- scaling limiter (Zhang & Shu 2010)
        The higher modes are multiplied by

            theta = 1                         if h(x_q) >= 0 at every point,
            theta = min_q h_bar/(h_bar-h_q)    otherwise (clipped to [0, 1]),

        with ``h_bar = coeff[:, 0, 0]`` preserved.  The filter is conservative
        (*the average depth is untouched*) and **idempotent where it is not
        needed**, which is what makes a still lake with a dry bank a discrete
        fixed point: its shoreline cell carries ``h = 0`` *exactly* at the dry
        node, and the old criterion ``h(x_q) > h_min`` used to rejected that
        wholly admissible trace and replace the cell by its cell average,
        zeroing the linear mode of ``h`` and with it ``eta = h + z_b = const``
        (measured before the fix: eta error 3.0e-2 m and spurious |u| up to
        0.54 m/s at P1, 0.50 m / 36 m/s at P2; see
        ``validation/swe_thacker/shoreline_rest_state_results.json``).

    ``well_balanced=False`` -- legacy replace-by-cell-average
        The historical action, kept bit-for-bit on the default path: a cell with
        any trace ``h(x_q) <= h_min`` is replaced by its average state.  It is
        *not* a positivity limiter in the Zhang-Shu sense -- it flattens the
        cell even when the trace is admissible -- and that flattening acts as
        front diffusion which the dry-bed dam-break verification relies on.
        Swapping it for the scaling limiter on this path sharpens the Ritter
        wetting front (mass error 3e-7 -> exact, spurious |u| 47 -> 5.9 m/s,
        relative L1 depth error 0.240% -> 0.224%) but moves the front measured
        with a ``h > 1e-6`` threshold from x = 3.00 to x = 2.85 against the
        analytic 3.13, so the existing dry-front gates have to be re-derived
        first.  That promotion is deliberately *not* part of this change.

    A cell can only be repaired when its cell-average depth is positive; in
    either mode a cell with ``h_bar <= h_min`` and any cell with non-finite data
    fall back to the average-only state, and ``solve_swe_1d`` raises if that
    average stays negative.
    """
    values = torch.einsum("cmv,qm->cqv", coeff, check_points)
    h_vals = values[..., 0]
    finite_cell = torch.isfinite(values).all(dim=2).all(dim=1)

    average_only = torch.zeros_like(coeff)
    average_only[:, 0] = coeff[:, 0]

    h_avg = coeff[:, 0, 0]
    feasible = finite_cell & (h_avg > h_min)

    if not well_balanced:
        # Legacy action: only cells whose worst trace is above h_min survive.
        valid_cell = ((h_vals > h_min) & torch.isfinite(values).all(dim=-1)).all(dim=1)
        return torch.where(valid_cell[:, None, None], coeff, average_only)

    # Scaling limiter: theta = 1 whenever no trace point is negative, so a
    # discretely resting state (h = 0 exactly at a dry node) is left alone.
    deficit = torch.clamp(-h_vals, min=0.0)                 # (ncells, nq)
    ratio = h_avg[:, None] / (h_avg[:, None] + deficit).clamp_min(1e-30)
    theta = torch.where(deficit > 0.0, ratio, torch.ones_like(deficit))
    theta = theta.min(dim=1).values.clamp(0.0, 1.0)
    theta = torch.where(feasible, theta, torch.zeros_like(theta))

    scaled = coeff * theta[:, None, None]
    scaled[:, 0] = coeff[:, 0]           # keep the cell average (conservation)
    return torch.where(feasible[:, None, None], scaled, average_only)


def solve_swe_1d(
    initial_h: Callable[[torch.Tensor], torch.Tensor],
    initial_u: Callable[[torch.Tensor], torch.Tensor],
    zb: Callable[[torch.Tensor], torch.Tensor],
    ncells: int = 100,
    degree: int = 1,
    final_time: float = 1.0,
    domain: tuple = (0.0, 1.0),
    cfl: float = 0.15,
    flux_kind: str = "hll",
    limiter: Optional[str] = None,
    tvb_constant: float = 1.0,
    limiter_stage: str = "final",
    rk_order: int = 3,
    h_min: float = 1e-12,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    periodic: bool = False,
    bc_left_flux: Optional[Callable[[float], torch.Tensor]] = None,
    bc_right_flux: Optional[Callable[[float], torch.Tensor]] = None,
    zb_slope: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    well_balanced: bool = False,
    hydrostatic: bool = True,
) -> tuple:
    """Solve the 1-D shallow water equations with DG.

    Parameters
    ----------
    initial_h : callable or tensor  h(x) at t=0
        A tensor of shape ``(ncells, degree + 1)`` is interpreted as the
        cell-local nodal values at the Gauss-Lobatto-Legendre nodes and is only
        accepted with ``well_balanced=True`` (see ``zb``).
    initial_u : callable or tensor  u(x) at t=0; same tensor convention
    zb : callable or tensor  bed elevation z_b(x); same tensor convention
        Cell-local nodal data is how a bed that jumps at a cell face is
        expressed: the left and right cells store their own face value.
    ncells : number of cells
    degree : polynomial degree (1 or 2)
    final_time : simulation end time
    domain : (x_left, x_right)
    cfl : CFL number
    flux_kind : "hll" or "lf"
    limiter : None, "tvb", or "weno"
    tvb_constant : TVB limiter constant M
    limiter_stage : "every" or "final"
    rk_order : SSP-RK order (2 or 3)
    h_min : tolerance of the positivity filter's feasibility test on the cell
        average.  The pointwise floor enforced by the filter is 0 (a dry
        trace is admissible, which is what a shoreline needs); a cell whose
        cell-average depth is not above ``h_min`` cannot be repaired by
        scaling and is replaced by its average state.
    periodic : periodic boundary conditions
    bc_left_flux : callable t -> (hu, hu^2 + g h^2 / 2) at x = x_left
        Exact boundary flux (verification mode). Overrides the default
        transmissive closure on the left boundary.
    bc_right_flux : callable t -> (hu, hu^2 + g h^2 / 2) at x = x_right
        Idem on the right boundary.
    zb_slope : callable  analytic dz_b/dx(x) sampled at the Gauss points.
        Opt-in: replaces the cell-constant bed slope of the legacy source by an
        exact quadrature.  Ignored when ``well_balanced=True`` (the derivative
        of the polynomial bed is used instead, which is the consistent choice).
    well_balanced : bool
        Exact lake-at-rest path.  The bed is represented by its degree-``p``
        LGL interpolant and the initial data are collocated at the *same* LGL
        nodes, so the discrete rest state ``h_h + z_h = const`` is
        representable to round-off; the momentum source uses the exact
        derivative of that polynomial bed.  For a smooth continuous bed this
        makes a still lake a discrete fixed point at machine precision for
        both P1 and P2 (the default L2-projected path leaves O(dx) drift at
        P1 and a non-decaying drift at P2 -- see
        ``docs/swe_well_balanced_exact.md``).

        Interfaces use the Audusse et al. (2004) hydrostatic reconstruction
        with ``z_b* = max(z_b,L, z_b,R)`` and ``h* = max(0, h + z_b - z_b*)``,
        plus the ``g/2 (h^2 - h*^2)`` momentum correction of each side, so a
        still lake is also a discrete fixed point when the bed is
        *discontinuous* between two cells and when a wet cell faces a dry one.
        On a continuous bed the reconstruction is the identity and the scheme
        is unchanged.
    hydrostatic : bool, default True
        Use the Audusse reconstruction of the well-balanced path.  ``False``
        reproduces the shared-face HLL flux of the pre-reconstruction scheme on
        the same per-cell bed, i.e. it is an audit/validation switch that shows
        the spurious momentum injected by a bed jump; production runs keep the
        default.

    Returns
    -------
    x : (ncells,) cell centres
    coeff : (ncells, degree+1, 2) modal coefficients [h, hu]
    zb_faces : (ncells, 2) bed elevation at left/right faces
    metadata : dict
    """
    if ncells < 4 or degree not in (1, 2):
        raise ValueError("ncells >= 4 and degree in {1, 2}")
    if rk_order not in (2, 3):
        raise ValueError("rk_order must be 2 or 3")
    if limiter_stage not in ("every", "final"):
        raise ValueError("limiter_stage must be 'every' or 'final'")
    if flux_kind not in ("hll", "lf"):
        raise ValueError("flux_kind must be 'hll' or 'lf'")

    lower, upper = domain
    if not upper > lower:
        raise ValueError("domain must be increasing")

    dev = torch.device(device)
    dx = (upper - lower) / ncells
    x = lower + (torch.arange(ncells, device=dev, dtype=dtype) + 0.5) * dx

    # Bed elevation at cell faces (a cell-local tensor bed is only meaningful
    # in the well-balanced branch, where the face values come from the bed
    # polynomials below).
    zb_left_faces = lower + torch.arange(ncells, device=dev, dtype=dtype) * dx
    zb_right_faces = zb_left_faces + dx
    if callable(zb):
        zb_l = zb(zb_left_faces)
        zb_r = zb(zb_right_faces)
    else:
        zb_l = zb_r = torch.zeros(ncells, device=dev, dtype=dtype)
    zb_jump = zb_r - zb_l  # [ncells]

    # Initial condition at quadrature points
    nodes, weights, phi, dphi, phi_left, phi_right = _operators(degree, dev, dtype)
    check_points = torch.cat((phi, phi_left[None, :], phi_right[None, :]), dim=0)

    node_x = x[:, None] + 0.5 * dx * nodes[None, :]
    zb_slope_quad = None if zb_slope is None else zb_slope(node_x)
    coeff = torch.zeros((ncells, degree + 1, 2), device=dev, dtype=dtype)
    bed_modal = None

    if well_balanced:
        # --- consistent (nodal) representation for exact lake-at-rest -------
        # Interpolate the bed and the initial data at the *same* LGL nodes.
        # Interpolation is linear, so h_h + z_h = eta_h is the interpolant of a
        # constant whenever the sampled h + z is constant -- the discrete rest
        # state then lies exactly in the DG space and both interface traces of
        # h agree, which is what makes the jump term of the numerical flux
        # vanish instead of leaving the O(dx) / O(1) drift of the L2 path.
        vinv, lgl_nodes = _lgl_transform(degree, dev, dtype)
        node_x_lgl = x[:, None] + 0.5 * dx * lgl_nodes[None, :]
        # Cell-local nodal data (tensor argument) lets a bed jump exactly at a
        # shared face: each cell then carries its own face value.
        cz = _nodal_values(zb, node_x_lgl, "zb") @ vinv.T  # bed modes
        # Face values of the polynomial bed (equal to zb at the faces, because
        # the interpolant passes through the LGL nodes -1 and +1).
        zb_l = cz @ phi_left
        zb_r = cz @ phi_right
        # Exact derivative of the *polynomial* bed at the Gauss points; using
        # the analytic dz_b/dx here instead would leave an O(dx^(p+1)) residual.
        zb_slope_quad = torch.einsum("cm,qm->cq", cz, dphi) * (2.0 / dx)
        h_nodes = _nodal_values(initial_h, node_x_lgl, "initial_h")
        u_nodes = _nodal_values(initial_u, node_x_lgl, "initial_u")
        coeff[:, :, 0] = h_nodes @ vinv.T
        coeff[:, :, 1] = (h_nodes * u_nodes) @ vinv.T
        # The face values above may now differ between neighbours; keep the
        # jump used by the legacy source branch consistent with them.
        zb_jump = zb_r - zb_l
        # Activate the Audusse hydrostatic reconstruction in _rhs.
        bed_modal = cz if hydrostatic else None
    else:
        if not (callable(initial_h) and callable(initial_u) and callable(zb)):
            raise ValueError(
                "tensor (cell-local nodal) initial data requires well_balanced=True")
        h0 = initial_h(node_x)
        u0 = initial_u(node_x)
        hu0 = h0 * u0
        # L2 projection onto modal basis
        for k in range(degree + 1):
            coeff[:, k, 0] = ((2 * k + 1) / 2.0) * torch.einsum(
                "cq,q,q->c", h0, weights, phi[:, k])
            coeff[:, k, 1] = ((2 * k + 1) / 2.0) * torch.einsum(
                "cq,q,q->c", hu0, weights, phi[:, k])

    # The scaling (well-balanced) action is active exactly where the Audusse
    # hydrostatic reconstruction is: that is the path whose contract is a
    # discrete rest state across wet/dry traces.  The default path keeps the
    # legacy action its dry-front verification was tuned against.
    hydrostatic_active = bed_modal is not None
    coeff = _positivity_filter(coeff, check_points, h_min, hydrostatic_active)

    limit_each_stage = limiter_stage == "every"
    time, steps, dt = 0.0, 0, 0.0
    stalled = False

    while time < final_time:
        values = torch.einsum("cmv,qm->cqv", coeff, phi)
        _, u_vals, c_vals = _primitive(values)
        max_speed = float(torch.max(torch.abs(u_vals) + c_vals).item())
        dt = min(cfl * dx / ((2 * degree + 1) * max(max_speed, 1e-12)), final_time - time)
        if dt < 1e-15:
            stalled = True
            break

        if rk_order == 3:
            k1 = _rhs(coeff, dx, weights, phi, dphi, phi_left, phi_right,
                      zb_jump, flux_kind, periodic, time,
                      bc_left_flux, bc_right_flux, zb_slope_quad, bed_modal)
            u1 = coeff + dt * k1
            if limit_each_stage:
                u1 = apply_limiter(u1, dx, kind=limiter, tvb_constant=tvb_constant)
            u1 = _positivity_filter(u1, check_points, h_min, hydrostatic_active)

            k2 = _rhs(u1, dx, weights, phi, dphi, phi_left, phi_right,
                      zb_jump, flux_kind, periodic, time + dt,
                      bc_left_flux, bc_right_flux, zb_slope_quad, bed_modal)
            u2 = 0.75 * coeff + 0.25 * (u1 + dt * k2)
            if limit_each_stage:
                u2 = apply_limiter(u2, dx, kind=limiter, tvb_constant=tvb_constant)
            u2 = _positivity_filter(u2, check_points, h_min, hydrostatic_active)

            k3 = _rhs(u2, dx, weights, phi, dphi, phi_left, phi_right,
                      zb_jump, flux_kind, periodic, time + 0.5 * dt,
                      bc_left_flux, bc_right_flux, zb_slope_quad, bed_modal)
            coeff = (1.0 / 3.0) * coeff + (2.0 / 3.0) * (u2 + dt * k3)
        else:
            k1 = _rhs(coeff, dx, weights, phi, dphi, phi_left, phi_right,
                      zb_jump, flux_kind, periodic, time,
                      bc_left_flux, bc_right_flux, zb_slope_quad, bed_modal)
            u1 = coeff + dt * k1
            if limit_each_stage:
                u1 = apply_limiter(u1, dx, kind=limiter, tvb_constant=tvb_constant)
            u1 = _positivity_filter(u1, check_points, h_min, hydrostatic_active)

            # Stage 2 must see the same time-dependent data as stage 1: the
            # analytic boundary flux and the time argument.
            k2 = _rhs(u1, dx, weights, phi, dphi, phi_left, phi_right,
                      zb_jump, flux_kind, periodic, time + dt,
                      bc_left_flux, bc_right_flux, zb_slope_quad, bed_modal)
            coeff = 0.5 * coeff + 0.5 * (u1 + dt * k2)

        if limiter_stage == "final" and limiter is not None:
            coeff = apply_limiter(coeff, dx, kind=limiter, tvb_constant=tvb_constant)
        coeff = _positivity_filter(coeff, check_points, h_min, hydrostatic_active)

        h_avg = coeff[:, 0, 0]
        if not torch.isfinite(coeff).all() or (h_avg < -h_min).any():
            raise FloatingPointError("Nonphysical SWE state; reduce CFL or check IC")

        time += dt
        steps += 1

    zb_faces = torch.stack((zb_l, zb_r), dim=1)
    metadata = {
        "ncells": ncells, "degree": degree, "final_time": final_time,
        "steps": steps, "dt": dt, "domain": (lower, upper), "cfl": cfl,
        "flux": flux_kind, "limiter": limiter, "rk_order": rk_order,
        "device": str(dev), "limiter_stage": limiter_stage,
        "exact_boundary_flux": bc_left_flux is not None or bc_right_flux is not None,
        "exact_bed_slope": zb_slope is not None,
        "well_balanced": well_balanced,
        "hydrostatic_reconstruction": bed_modal is not None,
        "bed_representation": ("nodal_lgl_polynomial" if well_balanced else "piecewise_linear"),
        "zb_modes": (cz if well_balanced else None),
        "time_reached": time,
        "reached_final_time": (not stalled) and time >= final_time - 1e-12,
        "dry_depth": DRY_DEPTH,
    }
    if not metadata["reached_final_time"]:
        warnings.warn(
            f"SWE time integration stopped at t={time:.6g} before the requested "
            f"final_time={final_time:.6g} (dt underflow at dt={dt:.3e}); the "
            "returned state is NOT the solution at final_time.",
            RuntimeWarning, stacklevel=2)
    return x, coeff, zb_faces, metadata


def reconstruct_fields(coeff: torch.Tensor, zb_faces: torch.Tensor,
                       zb_center: Optional[torch.Tensor] = None) -> tuple:
    """Reconstruct h, u, η at cell centres from modal coefficients.

    ``zb_center`` is the bed elevation used for the cell average of η.  The
    default is the trapezoidal mean ``(zb_left + zb_right) / 2`` of the
    piecewise-linear bed, which is *not* the cell average of a curved bed: for
    a degree-2 polynomial bed the two differ by ``O(dx^2)``, which shows up as
    a spurious ``O(dx^2)`` free-surface drift in a lake-at-rest check.  Passing
    the mode-0 coefficient of the polynomial bed (``metadata["zb_modes"][:, 0]``
    for a ``well_balanced=True`` run) removes that post-processing error.

    Returns (h_center, u_center, eta_center) each of shape (ncells,).
    """
    # Depth is non-negative by definition; the modal average of a dry cell can
    # sit at the -h_min positivity floor (-1e-12), which is reported as 0.
    h_avg = coeff[:, 0, 0].clamp_min(0.0)
    hu_avg = coeff[:, 0, 1]
    if zb_center is None:
        zb_center = 0.5 * (zb_faces[:, 0] + zb_faces[:, 1])
    u_center = torch.where(h_avg > DRY_DEPTH, hu_avg / h_avg.clamp_min(DRY_DEPTH),
                           torch.zeros_like(h_avg))
    eta_center = h_avg + zb_center
    return h_avg, u_center, eta_center


# ============================================================================
# Hydrostatic Well-Balanced Helper Functions
# ============================================================================

def check_hydrostatic_equilibrium(
    coeff: torch.Tensor,
    zb_faces: torch.Tensor,
    zb_center: Optional[torch.Tensor] = None,
    eta_expected: Optional[float] = None,
    tolerance: float = 1e-8,
) -> dict:
    """Check if a solution is in hydrostatic equilibrium.

    This function verifies that the solution satisfies the lake-at-rest condition:
    η = h + z_b = const and u = 0.

    Parameters
    ----------
    coeff : torch.Tensor
        Modal coefficients from solve_swe_1d
    zb_faces : torch.Tensor
        Bed elevation at cell faces
    zb_center : torch.Tensor, optional
        Bed elevation at cell centers (for curved beds)
    eta_expected : float, optional
        Expected constant value of η. If None, uses the mean of η.
    tolerance : float
        Tolerance for checking equilibrium (default: 1e-8)

    Returns
    -------
    dict
        Dictionary with keys:
        - 'is_hydrostatic': bool, True if equilibrium is satisfied
        - 'eta_error_max': float, max deviation of η from expected value
        - 'eta_error_l2': float, L2 norm of η deviation
        - 'u_max': float, max absolute velocity
        - 'u_l2': float, L2 norm of velocity
        - 'message': str, human-readable summary

    Examples
    --------
    >>> x, coeff, zb_faces, meta = solve_swe_1d(..., well_balanced=True)
    >>> result = check_hydrostatic_equilibrium(coeff, zb_faces, eta_expected=2.0)
    >>> if result['is_hydrostatic']:
    ...     print("Solution is in hydrostatic equilibrium")
    """
    h, u, eta = reconstruct_fields(coeff, zb_faces, zb_center)

    if eta_expected is None:
        eta_expected = float(eta.mean())

    eta_error = eta - eta_expected
    eta_error_max = float(eta_error.abs().max())
    eta_error_l2 = float(torch.sqrt(torch.mean(eta_error ** 2)))
    u_max = float(u.abs().max())
    u_l2 = float(torch.sqrt(torch.mean(u ** 2)))

    is_hydrostatic = (eta_error_max < tolerance) and (u_max < tolerance)

    if is_hydrostatic:
        message = (
            f"Hydrostatic equilibrium satisfied: "
            f"η error = {eta_error_max:.3e} m, u max = {u_max:.3e} m/s"
        )
    else:
        message = (
            f"NOT in hydrostatic equilibrium: "
            f"η error = {eta_error_max:.3e} m (tol={tolerance:.1e}), "
            f"u max = {u_max:.3e} m/s (tol={tolerance:.1e})"
        )

    return {
        "is_hydrostatic": is_hydrostatic,
        "eta_error_max": eta_error_max,
        "eta_error_l2": eta_error_l2,
        "u_max": u_max,
        "u_l2": u_l2,
        "eta_expected": eta_expected,
        "message": message,
    }


def create_hydrostatic_initial_conditions(
    zb: Callable[[torch.Tensor], torch.Tensor],
    eta0: float,
) -> tuple[Callable, Callable]:
    """Create initial conditions for hydrostatic equilibrium.

    Given a bed elevation function z_b(x) and a constant water surface level η₀,
    this function returns the corresponding depth and velocity initial conditions
    for a lake-at-rest state.

    Parameters
    ----------
    zb : callable
        Bed elevation function z_b(x) -> torch.Tensor
    eta0 : float
        Constant water surface level η₀

    Returns
    -------
    tuple[callable, callable]
        (initial_h, initial_u) where:
        - initial_h(x) = η₀ - z_b(x)
        - initial_u(x) = 0

    Examples
    --------
    >>> def zb(x):
    ...     return 0.5 * torch.sin(torch.pi * x)
    >>> initial_h, initial_u = create_hydrostatic_initial_conditions(zb, eta0=2.0)
    >>> x, coeff, zb_faces, meta = solve_swe_1d(
    ...     initial_h, initial_u, zb, well_balanced=True, ...)
    """
    def initial_h(x: torch.Tensor) -> torch.Tensor:
        return eta0 - zb(x)

    def initial_u(x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)

    return initial_h, initial_u


def compute_hydrostatic_pressure_force(
    coeff: torch.Tensor,
    zb_faces: torch.Tensor,
) -> torch.Tensor:
    """Compute the hydrostatic pressure force per unit width.

    The hydrostatic pressure force is:
        F_p = ∫₀ʰ ρg(h - z) dz = ½ρgh²

    This function computes the pressure force at each cell center.

    Parameters
    ----------
    coeff : torch.Tensor
        Modal coefficients from solve_swe_1d
    zb_faces : torch.Tensor
        Bed elevation at cell faces

    Returns
    -------
    torch.Tensor
        Pressure force per unit width at each cell center (shape: [ncells])
        Units: N/m (assuming ρ = 1000 kg/m³)

    Notes
    -----
    This is useful for computing forces on structures or checking momentum balance.
    """
    h, u, eta = reconstruct_fields(coeff, zb_faces)
    # F_p = ½ρgh² (per unit width, with ρ = 1000 kg/m³)
    rho = 1000.0  # kg/m³
    pressure_force = 0.5 * rho * GRAVITY * h ** 2
    return pressure_force


def estimate_well_balanced_error_bound(
    ncells: int,
    degree: int,
    domain_length: float,
    bed_curvature: float = 1.0,
    floor: Optional[float] = None,
    reference_depth: float = 1.0,
) -> dict:
    """Estimate the error bound for hydrostatic well-balanced property.

    For the nodal LGL polynomial bed representation, the error in preserving
    hydrostatic equilibrium depends on:
    1. The polynomial degree p
    2. The mesh spacing dx
    3. The curvature of the bed

    This function provides a theoretical error estimate based on the approximation
    theory for polynomial interpolation.

    Parameters
    ----------
    ncells : int
        Number of cells
    degree : int
        Polynomial degree (1 or 2)
    domain_length : float
        Length of the computational domain
    bed_curvature : float
        Characteristic curvature of the bed (d²z_b/dx²)
    floor : float, optional
        Optional round-off floor, in metres, below which the discrete
        equilibrium residual cannot go.  ``None`` (default) means *no* floor is
        imposed and the returned bound is the pure representation estimate.
        This used to be a hard-coded ``1e-10`` applied to every call, which
        silently reported ``1e-10`` for a bed whose estimate is ``2e-3``
        (``ncells=8``, P1, ``bed_curvature=1``) and made the bound useless as
        a refinement indicator.  Pass the measured round-off level of your
        configuration explicitly if you want the floor to enter the bound.
    reference_depth : float
        Reference water depth, in metres, used to turn a free-surface error
        into a velocity scale, ``u ~ η sqrt(g / reference_depth)`` (the
        linearised shallow-water relation).  Default ``1.0`` m.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'dx': float, mesh spacing
        - 'eta_error_estimate': float, representation estimate for η
          (``bed_curvature * dx**(p+1) / c_p``), never clamped
        - 'expected_eta_error': float, ``max(eta_error_estimate, floor)``
        - 'floor_applied': bool, True if ``floor`` actually bound the estimate
        - 'expected_u_error': float, linearised velocity scale
          ``expected_eta_error * sqrt(g / reference_depth)``
        - 'convergence_rate': float, theoretical convergence rate
        - 'notes': str, explanation of the estimate

    Notes
    -----
    For well_balanced=True with nodal LGL representation:
    - P1: error ~ O(dx²) for quadratic beds
    - P2: error ~ O(dx³) for cubic beds
    - For a bed that is a polynomial of degree ≤ p the representation is
      exact and this estimate is zero; the residual actually observed in a
      solve then drops to the round-off floor of the scheme (measured in
      ``docs/tensordg-swe-hydrostatic-wellbalanced.md``, ~1e-10 m for P1).
      That floor is a property of the arithmetic, not of this estimate, so it
      is *not* baked into the returned bound: pass ``floor=`` if you want it.
    """
    dx = domain_length / ncells

    # Theoretical error estimates
    # For polynomial bed of degree ≤ p, the representation is exact
    # For higher-degree beds, the error is O(dx^(p+1))
    if degree == 1:
        convergence_rate = 2.0
        expected_eta_error = bed_curvature * dx ** 2 / 8.0
    elif degree == 2:
        convergence_rate = 3.0
        expected_eta_error = bed_curvature * dx ** 3 / 24.0
    else:
        convergence_rate = float(degree + 1)
        expected_eta_error = bed_curvature * dx ** (degree + 1) / (2 ** degree * math.factorial(degree + 1))

    eta_error_estimate = expected_eta_error
    if floor is None:
        floor_applied = False
        expected_eta_error = eta_error_estimate
    else:
        floor_applied = floor > eta_error_estimate
        expected_eta_error = max(eta_error_estimate, float(floor))

    if reference_depth <= 0.0:
        raise ValueError(
            f"reference_depth must be positive, got {reference_depth!r}")
    expected_u_error = expected_eta_error * math.sqrt(GRAVITY / reference_depth)

    notes = (
        f"Representation estimate only (no round-off floor imposed): the bed is "
        f"interpolated by a degree-{degree} LGL polynomial, so the η error scales "
        f"as O(dx^{degree + 1}) with dx={dx:.3e} m and a bed curvature of "
        f"{bed_curvature:.3e}; eta_error_estimate = {eta_error_estimate:.3e} m. "
        f"For a bed that is a polynomial of degree ≤ {degree} the estimate is 0 "
        f"and the observed residual is the round-off floor of the scheme "
        f"(~1e-10 m for P1, ~1e-9 m/s in u), which is a property of the "
        f"arithmetic and is not included unless floor= is passed. "
        f"expected_u_error is the linearised relation u ~ η sqrt(g / h_ref) with "
        f"h_ref = {reference_depth:g} m, i.e. a velocity *scale*, not a bound "
        f"derived from the momentum equation."
    )

    return {
        "dx": dx,
        "eta_error_estimate": eta_error_estimate,
        "expected_eta_error": expected_eta_error,
        "floor_applied": floor_applied,
        "expected_u_error": expected_u_error,
        "convergence_rate": convergence_rate,
        "notes": notes,
    }


def validate_well_balanced_setup(
    initial_h: Callable,
    initial_u: Callable,
    zb: Callable,
    test_points: Optional[torch.Tensor] = None,
    tolerance: float = 1e-10,
) -> dict:
    """Validate that initial conditions are consistent with hydrostatic equilibrium.

    This function checks whether the provided initial conditions satisfy the
    lake-at-rest condition η = h + z_b = const and u = 0 at a set of test points.

    Parameters
    ----------
    initial_h : callable
        Initial depth function h(x)
    initial_u : callable
        Initial velocity function u(x)
    zb : callable
        Bed elevation function z_b(x)
    test_points : torch.Tensor, optional
        Points at which to check equilibrium. If None, uses 100 points in [0, 1].
    tolerance : float
        Tolerance for checking equilibrium, in metres for η and in m/s for u.
        Both criteria are supremum norms, so the same number means the same
        thing here as in :func:`check_hydrostatic_equilibrium`.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'is_valid': bool, True if conditions are consistent
        - 'eta_deviation_max': float, sup-norm |η - mean(η)| in metres; this is
          the quantity compared against ``tolerance``
        - 'eta_variance': float, variance of η across test points (kept for
          backwards compatibility; it is *not* the acceptance criterion — a
          variance has units of m², so testing it against a 1e-10 "tolerance"
          accepted an η drift of 1e-5 m, 1e5x the stated tolerance)
        - 'eta_mean': float, mean of η
        - 'u_max': float, max absolute velocity
        - 'message': str, human-readable summary

    Examples
    --------
    >>> def zb(x): return 0.3 * torch.cos(2 * torch.pi * x)
    >>> initial_h, initial_u = create_hydrostatic_initial_conditions(zb, 2.0)
    >>> result = validate_well_balanced_setup(initial_h, initial_u, zb)
    >>> if result['is_valid']:
    ...     print("Initial conditions are valid for hydrostatic equilibrium")
    """
    if test_points is None:
        test_points = torch.linspace(0.0, 1.0, 100, dtype=torch.float64)

    h_vals = initial_h(test_points)
    u_vals = initial_u(test_points)
    zb_vals = zb(test_points)
    eta_vals = h_vals + zb_vals

    eta_mean = float(eta_vals.mean())
    eta_variance = float(eta_vals.var())
    # Sup-norm deviation: the acceptance criterion.  Comparing the *variance*
    # against a length tolerance mixes units (m² vs m) and is quadratic in the
    # amplitude, so a small drift is squared away (1e-5 m drift -> variance
    # 5e-11 < 1e-10 "tolerance" -> accepted).
    eta_deviation_max = float((eta_vals - eta_mean).abs().max())
    u_max = float(u_vals.abs().max())

    is_valid = (eta_deviation_max < tolerance) and (u_max < tolerance)

    if is_valid:
        message = (
            f"Initial conditions are valid for hydrostatic equilibrium: "
            f"η = {eta_mean:.6f} ± {eta_deviation_max:.3e} (max deviation), "
            f"u max = {u_max:.3e}"
        )
    else:
        message = (
            f"Initial conditions are NOT valid for hydrostatic equilibrium: "
            f"max |η - mean| = {eta_deviation_max:.3e} m (tol={tolerance:.1e}), "
            f"u max = {u_max:.3e} (tol={tolerance:.1e})"
        )

    return {
        "is_valid": is_valid,
        "eta_mean": eta_mean,
        "eta_deviation_max": eta_deviation_max,
        "eta_variance": eta_variance,
        "u_max": u_max,
        "message": message,
    }
