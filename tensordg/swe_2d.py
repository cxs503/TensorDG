"""Two-dimensional shallow water equations with tensor-product DG(P1/P2).

Solves the 2-D SWE system with bed slope source term.

Conservative variables U = [h, hu, hv].
"""
from __future__ import annotations

from typing import Callable, Optional

from .stepping import step_metadata

import numpy as np
import torch

GRAVITY = 9.81

# Depth below which a trace is treated as carrying no momentum.  The value is
# shared with the 1-D solver (``swe_1d.DRY_DEPTH``) so that the wet/dry
# conventions of the two solvers agree.
DRY_DEPTH = 1e-6


def _lgl_transform_2d(degree: int, device, dtype):
    """Nodal -> modal transform on the degree-``p`` Gauss-Lobatto-Legendre grid.

    Returns ``(V^{-1}, nodes)`` with ``V[j, k] = P_k(xi_j)`` at the ``p + 1``
    LGL nodes (both endpoints included), exactly as ``swe_1d._lgl_transform``.
    Because the nodes include ``xi = +-1`` the interpolant reproduces the nodal
    values at the cell faces, which is what makes the polynomial bed of the
    well-balanced path agree with the bed stored on the faces.
    """
    nodes_np = (np.array([-1.0, 1.0]) if degree == 1
                else np.array([-1.0, 0.0, 1.0]))
    vander = np.stack(
        [np.polynomial.legendre.legval(nodes_np, e) for e in np.eye(degree + 1)],
        axis=1)
    vinv = torch.linalg.inv(
        torch.as_tensor(vander, device=device, dtype=dtype))
    return vinv, torch.as_tensor(nodes_np, device=device, dtype=dtype)


def physical_flux_2d(state: torch.Tensor, axis: int) -> torch.Tensor:
    """Physical flux for 2-D SWE along given axis."""
    h = state[..., 0].clamp_min(0.0)
    hu = state[..., 1]
    hv = state[..., 2]
    p = 0.5 * GRAVITY * h ** 2

    if axis == 0:
        return torch.stack((hu, hu ** 2 / h.clamp_min(1e-12) + p, hu * hv / h.clamp_min(1e-12)), dim=-1)
    else:
        return torch.stack((hv, hu * hv / h.clamp_min(1e-12), hv ** 2 / h.clamp_min(1e-12) + p), dim=-1)


def sound_speed_2d(state: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(GRAVITY * state[..., 0].clamp_min(1e-12))


def hll_flux_2d(ul: torch.Tensor, ur: torch.Tensor, axis: int) -> torch.Tensor:
    """HLL flux for 2-D SWE."""
    h_l = ul[..., 0].clamp_min(1e-12)
    h_r = ur[..., 0].clamp_min(1e-12)

    normal_idx = 1 if axis == 0 else 2
    u_n_l = ul[..., normal_idx] / h_l
    u_n_r = ur[..., normal_idx] / h_r

    c_l = sound_speed_2d(ul)
    c_r = sound_speed_2d(ur)

    s_l = torch.minimum(u_n_l - c_l, u_n_r - c_r)
    s_r = torch.maximum(u_n_l + c_l, u_n_r + c_r)

    fl = physical_flux_2d(ul, axis)
    fr = physical_flux_2d(ur, axis)

    denom = s_r - s_l
    safe_denom = torch.where(denom.abs() < 1e-30, torch.full_like(denom, 1e-30), denom)
    flux = (s_r[..., None] * fl - s_l[..., None] * fr
            + s_l[..., None] * s_r[..., None] * (ur - ul)) / safe_denom[..., None]

    flux = torch.where((s_l >= 0.0)[..., None], fl, flux)
    flux = torch.where((s_r <= 0.0)[..., None], fr, flux)
    return flux


def legendre_operators_2d(degree: int, device="cpu", dtype=torch.float64):
    """Return 1-D Legendre operators for tensor-product DG."""
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


def _hydrostatic_flux_2d(coeff, bed_modal, phi, phi_left, phi_right, weights,
                         axis: int = 0, periodic: bool = False):
    """Audusse et al. (2004) hydrostatic reconstruction + HLL flux, per face.

    ``bed_modal`` is the tensor-product bed polynomial ``(nx, ny, p+1, p+1)``.
    For the face shared by cells ``i`` and ``i+1`` the two one-sided traces are
    reconstructed onto the higher of the two face beds,

        z_b* = max(z_b,iR, z_b,i+1,L),  h*_m = max(0, h_m + z_b,m - z_b*),

    the HLL flux is evaluated with the reconstructed states and the pressure
    correction ``g/2 (h_m^2 - h*_m^2)`` is added to the *normal* momentum
    component of that side.  The tangential velocity is carried unchanged, as
    in the 1-D solver and in Audusse et al.

    On a continuous bed the reconstruction is the identity, so the returned
    one-sided fluxes coincide with the shared-face HLL flux of the default
    path; it only changes the result across a bed jump or a wet/dry interface.

    Returns ``(flux_right, flux_left)``, each ``(nx, ny, nmodes, nvar)`` and
    already projected onto the tangential modes, so it can be used exactly like
    the ``flux_right_x``/``flux_left_x`` arrays of the default path.
    """
    normal = 1 if axis == 0 else 2

    if axis == 0:
        tr_r = torch.einsum("xyklv,k,bl->xybv", coeff, phi_right, phi)
        tr_l = torch.einsum("xyklv,k,bl->xybv", coeff, phi_left, phi)
        b_r = torch.einsum("xykl,k,bl->xyb", bed_modal, phi_right, phi)
        b_l = torch.einsum("xykl,k,bl->xyb", bed_modal, phi_left, phi)
        roll_dim = 0
    else:
        tr_r = torch.einsum("xyklv,l,ak->xyav", coeff, phi_right, phi)
        tr_l = torch.einsum("xyklv,l,ak->xyav", coeff, phi_left, phi)
        b_r = torch.einsum("xykl,l,ak->xya", bed_modal, phi_right, phi)
        b_l = torch.einsum("xykl,l,ak->xya", bed_modal, phi_left, phi)
        roll_dim = 1

    def project(flux):
        if axis == 0:
            return torch.einsum("xybv,b,bl->xylv", flux, weights, phi)
        return torch.einsum("xyav,a,ak->xykv", flux, weights, phi)

    nb_tr = torch.roll(tr_l, shifts=-1, dims=roll_dim)
    nb_b = torch.roll(b_l, shifts=-1, dims=roll_dim)
    if not periodic:
        last = [slice(None)] * 3
        last[roll_dim] = -1
        nb_tr[tuple(last)] = tr_r[tuple(last)]
        nb_b[tuple(last)] = b_r[tuple(last)]

    def side(trace, bed, other_bed):
        h = trace[..., 0].clamp_min(0.0)
        u_n = torch.where(h > DRY_DEPTH, trace[..., normal] / h.clamp_min(DRY_DEPTH),
                          torch.zeros_like(h))
        tang = 3 - normal  # 2 if normal == 1 else 1
        u_t = torch.where(h > DRY_DEPTH, trace[..., tang] / h.clamp_min(DRY_DEPTH),
                          torch.zeros_like(h))
        star = torch.maximum(bed, other_bed)
        h_star = (h + bed - star).clamp_min(0.0)
        # Conservative variables are (h, hu, hv): the reconstructed normal
        # momentum goes to component 1 for axis 0 and to component 2 for
        # axis 1; the tangential velocity is carried across unchanged.
        mom_n = h_star * u_n
        mom_t = h_star * u_t
        state = (torch.stack((h_star, mom_n, mom_t), dim=-1) if axis == 0
                 else torch.stack((h_star, mom_t, mom_n), dim=-1))
        corr = 0.5 * GRAVITY * (h ** 2 - h_star ** 2)
        return state, corr

    state_a, corr_a = side(tr_r, b_r, nb_b)
    state_b, corr_b = side(nb_tr, nb_b, b_r)

    flux = hll_flux_2d(state_a, state_b, axis)
    flux_a = flux.clone()
    flux_a[..., normal] = flux_a[..., normal] + corr_a
    flux_b = flux.clone()
    flux_b[..., normal] = flux_b[..., normal] + corr_b

    flux_right = project(flux_a)
    flux_left = torch.roll(project(flux_b), shifts=1, dims=roll_dim)
    if not periodic:
        # First cell's left face: no left neighbour, the transmissive closure
        # of the default path is the physical flux of the cell's own trace
        # (equivalently the reconstruction with z_b* = z_b,L, for which the
        # correction vanishes identically).
        first = [slice(None)] * 3
        first[roll_dim] = 0
        flux_left[tuple(first)] = project(
            hll_flux_2d(tr_l, tr_l, axis))[tuple(first)]
    return flux_right, flux_left


def _rhs_2d(coeff, dx, dy, weights, phi, dphi, phi_left, phi_right,
            zb_jump_x, zb_jump_y, periodic=(False, False),
            bed_modal=None, zb_slope_quad_x=None, zb_slope_quad_y=None):
    """DG residual for 2-D SWE on Cartesian mesh.

    coeff: (nx, ny, nmodes, nmodes, 3)

    ``bed_modal`` (``(nx, ny, p+1, p+1)``) switches the interface fluxes to the
    Audusse hydrostatic reconstruction of the well-balanced path;
    ``zb_slope_quad_x``/``zb_slope_quad_y`` (``(nx, ny, nq, nq)``) replace the
    cell-constant bed slope of the default source term by the exact derivative
    of that bed polynomial sampled at the same Gauss points as the solution.
    Both are ``None`` by default, which reproduces the pre-existing scheme bit
    for bit.
    """
    nx, ny, nmodes, _, nvar = coeff.shape

    # Evaluate at tensor-product quadrature
    at_quad = torch.einsum("xyklv,ak,bl->xyabv", coeff, phi, phi)
    h_quad = at_quad[..., 0]

    # Volume integrals
    vol_x = torch.einsum("xyabv,a,b,ak,bl->xyklv",
                         physical_flux_2d(at_quad, 0), weights, weights, dphi, phi)
    vol_y = torch.einsum("xyabv,a,b,ak,bl->xyklv",
                         physical_flux_2d(at_quad, 1), weights, weights, phi, dphi)

    # X-direction fluxes: trace at x-faces (contract x-modes, evaluate at the
    # y-quadrature points, so that the interface flux can be projected onto the
    # y-modes afterwards).  trace[x, y, b, v] with b = y-quadrature index.
    trace_right_x = torch.einsum("xyklv,k,bl->xybv", coeff, phi_right, phi)
    trace_left_x = torch.einsum("xyklv,k,bl->xybv", coeff, phi_left, phi)

    if bed_modal is None:
        neighbour_x = torch.roll(trace_left_x, shifts=-1, dims=0)
        if not periodic[0]:
            neighbour_x[-1] = trace_right_x[-1]

        iface_x = hll_flux_2d(trace_right_x, neighbour_x, 0)
        # project the x-interface flux onto the y-modes: [nx, ny, b, v] -> [nx, ny, l, v]
        flux_right_x = torch.einsum("xybv,b,bl->xylv", iface_x, weights, phi)
        flux_left_x = torch.roll(flux_right_x, shifts=1, dims=0)
        if not periodic[0]:
            flux_left_x[0] = torch.einsum("ybv,b,bl->ylv",
                                           hll_flux_2d(trace_left_x[:1], trace_left_x[:1], 0)[0],
                                           weights, phi)
    else:
        # Well-balanced branch: one-sided hydrostatic reconstruction.
        flux_right_x, flux_left_x = _hydrostatic_flux_2d(
            coeff, bed_modal, phi, phi_left, phi_right, weights,
            axis=0, periodic=periodic[0])

    # Y-direction fluxes: trace at y-faces.  The trace is the pointwise state
    # at (x-quadrature node a, eta = +-1), i.e. *both* the y-mode AND the x-mode
    # are contracted: sigma_{k,l} c_kl phi_k(x_a) phi_l(+-1) -> [x, y, a, v].
    # Keeping the x-mode index in the output (an earlier "xykav" contraction,
    # mission m6386) multiplied every mode by phi_k(x_a) instead of summing the
    # x-modes, which injected a spurious y-momentum source in the cross mode
    # (1,1) of an otherwise y-invariant state and made the solver blow up in a
    # single time step for any unsteady flow.
    trace_right_y = torch.einsum("xyklv,l,ak->xyav", coeff, phi_right, phi)
    trace_left_y = torch.einsum("xyklv,l,ak->xyav", coeff, phi_left, phi)

    if bed_modal is None:
        neighbour_y = torch.roll(trace_left_y, shifts=-1, dims=1)
        if not periodic[1]:
            neighbour_y[:, -1] = trace_right_y[:, -1]

        iface_y = hll_flux_2d(trace_right_y, neighbour_y, 1)
        # project the y-interface flux onto the x-modes: [nx, ny, a, v] -> [nx, ny, k, v]
        flux_right_y = torch.einsum("xyav,a,ak->xykv", iface_y, weights, phi)
        flux_left_y = torch.roll(flux_right_y, shifts=1, dims=1)
        if not periodic[1]:
            flux_left_y[:, 0] = torch.einsum("xav,a,ak->xkv",
                                              hll_flux_2d(trace_left_y[:, :1], trace_left_y[:, :1], 1)[:, 0],
                                              weights, phi)
    else:
        flux_right_y, flux_left_y = _hydrostatic_flux_2d(
            coeff, bed_modal, phi, phi_left, phi_right, weights,
            axis=1, periodic=periodic[1])

    out = torch.empty_like(coeff)
    for k in range(nmodes):
        for l in range(nmodes):
            # d c_kl/dt = (2k+1)(2l+1)/(2 dx) * [x-terms]
            #          + (2k+1)(2l+1)/(2 dy) * [y-terms]
            # The x-terms are already integrated over eta (y-projection) and the
            # y-terms over xi, hence the separate 1/dx and 1/dy scalings: the
            # weak-form factor for direction d is (2k+1)(2l+1)/(2 d_d).
            fx = (2 * k + 1) * (2 * l + 1) / (2.0 * dx)
            fy = (2 * k + 1) * (2 * l + 1) / (2.0 * dy)
            out[:, :, k, l] = fx * (
                vol_x[:, :, k, l] - flux_right_x[:, :, l] * phi_right[k]
                + flux_left_x[:, :, l] * phi_left[k]
            ) + fy * (
                vol_y[:, :, k, l] - flux_right_y[:, :, k] * phi_right[l]
                + flux_left_y[:, :, k] * phi_left[l]
            )
            # Weak-form bed-slope source: the (2k+1)/dx prefactor and the
            # 2/(2k+1) modal moment cancel, leaving -g dbdx * c_kl^h.
            if zb_slope_quad_x is None:
                src_x = -GRAVITY * (zb_jump_x / dx) * coeff[:, :, k, l, 0]
                src_y = -GRAVITY * (zb_jump_y / dy) * coeff[:, :, k, l, 0]
            else:
                # Well-balanced branch: exact derivative of the bed polynomial
                # at the same Gauss points as h.  Weak form of -g h dz_b/dx on
                # mode (k, l) in the reference cell:
                #   -(2k+1)(2l+1)/4 g sum_{a,b} w_a w_b h dz_b/dx phi_k phi_l
                src_w = (2 * k + 1) * (2 * l + 1) / 4.0
                src_x = -src_w * GRAVITY * torch.einsum(
                    "xyab,a,b,a,b->xy", h_quad * zb_slope_quad_x,
                    weights, weights, phi[:, k], phi[:, l])
                src_y = -src_w * GRAVITY * torch.einsum(
                    "xyab,a,b,a,b->xy", h_quad * zb_slope_quad_y,
                    weights, weights, phi[:, k], phi[:, l])
            out[:, :, k, l, 1] += src_x
            out[:, :, k, l, 2] += src_y

    return out


def _positivity_filter_2d(coeff, check_points, h_min=1e-12):
    """Ensure h ≥ h_min at all quadrature points."""
    values = torch.einsum("xyklv,ak,bl->xyabv", coeff, check_points, check_points)
    h_vals = values[..., 0]
    valid = (h_vals > h_min) & torch.isfinite(values).all(dim=-1)
    valid_cell = valid.all(dim=(2, 3))

    average_only = torch.zeros_like(coeff)
    average_only[:, :, 0, 0] = coeff[:, :, 0, 0]

    return torch.where(valid_cell[:, :, None, None, None], coeff, average_only)


def solve_swe_2d(
    initial_h: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    initial_u: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    initial_v: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    zb: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    nx: int = 32,
    ny: int = 32,
    degree: int = 1,
    final_time: float = 1.0,
    domain: tuple = ((0.0, 1.0), (0.0, 1.0)),
    cfl: float = 0.15,
    limiter: Optional[str] = None,
    tvb_constant: float = 1.0,
    h_min: float = 1e-12,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    periodic: tuple = (False, False),
    well_balanced: bool = False,
) -> tuple:
    """Solve the 2-D shallow water equations with tensor-product DG.

    Parameters
    ----------
    well_balanced : bool, default False
        Hydrostatic well-balanced path.  The bed is represented by its
        degree-``p`` tensor-product Gauss-Lobatto-Legendre interpolant and the
        initial data are collocated at the *same* LGL nodes, so a lake at rest
        (``h + z_b = const``, ``u = v = 0``) is exactly representable.  The
        interface fluxes use the Audusse hydrostatic reconstruction and the
        momentum source uses the exact derivative of that bed polynomial at
        the Gauss points of the solution, so the flux divergence and the
        source cancel pointwise and the discrete rest state is preserved to
        round-off -- on a *curved* bed as well, where the default
        cell-constant slope leaves a spurious current.

        Default ``False`` reproduces the pre-existing scheme bit for bit.
    """
    if nx < 4 or ny < 4 or degree not in (1, 2):
        raise ValueError("nx, ny >= 4 and degree in {1, 2}")
    if limiter not in (None, "tvb", "weno"):
        raise ValueError("limiter must be None, 'tvb' or 'weno'")

    if limiter is None:
        def _limit(state):
            return state
    else:
        # The tensor-product limiters are axis-generic in the number of
        # conserved variables (they only touch the modal axes), so the 2-D
        # Euler limiters apply to [h, hu, hv] unchanged.  Before mission
        # m6386 the ``limiter`` argument was accepted and silently dropped,
        # so a user asking for a limiter got an unlimited run.
        from .euler_dg_2d import apply_tvb_limiter_2d, apply_weno_limiter_2d

        if limiter == "tvb":
            def _limit(state):
                return apply_tvb_limiter_2d(
                    state, dx, dy, tvb_constant, periodic)
        else:
            def _limit(state):
                return apply_weno_limiter_2d(
                    state, dx, dy, tvb_constant, periodic)

    dev = torch.device(device)
    (xl, xr), (yl, yr) = domain
    dx = (xr - xl) / nx
    dy = (yr - yl) / ny

    x = xl + (torch.arange(nx, device=dev, dtype=dtype) + 0.5) * dx
    y = yl + (torch.arange(ny, device=dev, dtype=dtype) + 0.5) * dy

    # Bed elevation on the cell faces.  Both arguments must be shaped
    # (nx, 1) / (1, ny) so that the result is a genuine (nx, ny) array;
    # broadcasting (nx,) against (ny, 1) silently transposes the bed.
    x_faces = xl + torch.arange(nx + 1, device=dev, dtype=dtype) * dx
    y_faces = yl + torch.arange(ny + 1, device=dev, dtype=dtype) * dy
    zb_is_nodal = torch.is_tensor(zb)
    if zb_is_nodal:
        # Cell-local nodal bed: the face values come from the polynomial bed
        # (see below), since there is no single-valued callable to sample.
        zb_lx = zb_rx = zb_ly = zb_ry = None
        zb_jump_x = zb_jump_y = None
    else:
        zb_lx = zb(x_faces[:-1, None], y[None, :])
        zb_rx = zb(x_faces[1:, None], y[None, :])
        zb_ly = zb(x[:, None], y_faces[:-1][None, :])
        zb_ry = zb(x[:, None], y_faces[1:][None, :])

        zb_jump_x = zb_rx - zb_lx
        zb_jump_y = zb_ry - zb_ly

    nodes, weights, phi, dphi, phi_left, phi_right = legendre_operators_2d(degree, dev, dtype)
    check_points = torch.cat((phi, phi_left[None, :], phi_right[None, :]), dim=0)

    nmodes = degree + 1
    coeff = torch.zeros((nx, ny, nmodes, nmodes, 3), device=dev, dtype=dtype)
    bed_modal = None
    zb_slope_quad_x = None
    zb_slope_quad_y = None

    if well_balanced:
        # Nodal LGL collocation: the bed polynomial and the initial state share
        # one set of nodes, so h_h + z_h = const is reproducible exactly.
        vinv, lgl_nodes = _lgl_transform_2d(degree, dev, dtype)
        xq_lgl = x[:, None] + 0.5 * dx * lgl_nodes[None, :]
        yq_lgl = y[:, None] + 0.5 * dy * lgl_nodes[None, :]

        def _nodal(data, name):
            """Nodal LGL data: a callable sampled here, or cell-local values.

            A tensor of shape ``(nx, ny, degree + 1, degree + 1)`` is taken as
            the cell-local values at the LGL nodes.  That is the only way to
            express a bed or an interface that *jumps between two cells*: a
            single callable is evaluated at one shared coordinate and therefore
            smears the jump over the two cells.
            """
            if torch.is_tensor(data):
                arr = data.to(device=dev, dtype=dtype)
                expected = (nx, ny, nmodes, nmodes)
                if tuple(arr.shape) != expected:
                    raise ValueError(
                        f"{name} given as a tensor must have shape {expected} "
                        f"(cell-local LGL nodal values), got {tuple(arr.shape)}")
                return arr
            return data(xq_lgl[:, None, :, None], yq_lgl[None, :, None, :])

        hL = _nodal(initial_h, "initial_h")
        uL = _nodal(initial_u, "initial_u")
        vL = _nodal(initial_v, "initial_v")
        zbL = _nodal(zb, "zb")

        def _to_modal(nodal):
            # c_ij = sum_{m,n} V^{-1}_{im} f_{mn} (V^{-1})_{jn}
            return torch.einsum("im,xymn,jn->xyij", vinv, nodal, vinv)

        coeff[:, :, :, :, 0] = _to_modal(hL)
        coeff[:, :, :, :, 1] = _to_modal(hL * uL)
        coeff[:, :, :, :, 2] = _to_modal(hL * vL)
        bed_modal = _to_modal(zbL)
        if zb_is_nodal:
            # The interpolant reproduces the nodal values at ``xi = +-1``, so
            # these are the cell-local face beds (left/right cell values are
            # allowed to differ across a face: that is a bed jump).
            zb_lx = torch.einsum("xyk,k->xy", bed_modal[:, :, :, 0], phi_left)
            zb_rx = torch.einsum("xyk,k->xy", bed_modal[:, :, :, 0], phi_right)
            zb_ly = torch.einsum("xyl,l->xy", bed_modal[:, :, 0, :], phi_left)
            zb_ry = torch.einsum("xyl,l->xy", bed_modal[:, :, 0, :], phi_right)
        # Exact bed derivative at the Gauss points of the solution.
        zb_slope_quad_x = (2.0 / dx) * torch.einsum(
            "xykl,ak,bl->xyab", bed_modal, dphi, phi)
        zb_slope_quad_y = (2.0 / dy) * torch.einsum(
            "xykl,ak,bl->xyab", bed_modal, phi, dphi)
    else:
        for _name, _data in (("initial_h", initial_h), ("initial_u", initial_u),
                             ("initial_v", initial_v), ("zb", zb)):
            if torch.is_tensor(_data):
                raise ValueError(
                    f"{_name} was given as cell-local nodal data; that is only "
                    "supported with well_balanced=True, whose collocation makes "
                    "the nodal values the degrees of freedom.")
        xq = x[:, None] + 0.5 * dx * nodes[None, :]
        yq = y[:, None] + 0.5 * dy * nodes[None, :]

        h0 = initial_h(xq[:, None, :, None], yq[None, :, None, :])
        u0 = initial_u(xq[:, None, :, None], yq[None, :, None, :])
        v0 = initial_v(xq[:, None, :, None], yq[None, :, None, :])
        hu0 = h0 * u0
        hv0 = h0 * v0

        for k in range(nmodes):
            for l in range(nmodes):
                # L2 projection onto the tensor-product Legendre basis
                #   c_kl = ((2k+1)(2l+1)/4) * int f phi_k phi_l  (both basis
                #   functions must be present, otherwise every mode receives the
                #   plain cell average scaled by the (2k+1)(2l+1)/4 factor).
                prefac = ((2 * k + 1) / 2.0) * ((2 * l + 1) / 2.0)
                phi_k, phi_l = phi[:, k], phi[:, l]
                coeff[:, :, k, l, 0] = prefac * torch.einsum(
                    "xyab,a,b,a,b->xy", h0, weights, weights, phi_k, phi_l)
                coeff[:, :, k, l, 1] = prefac * torch.einsum(
                    "xyab,a,b,a,b->xy", hu0, weights, weights, phi_k, phi_l)
                coeff[:, :, k, l, 2] = prefac * torch.einsum(
                    "xyab,a,b,a,b->xy", hv0, weights, weights, phi_k, phi_l)

    coeff = _positivity_filter_2d(_limit(coeff), check_points, h_min)

    time, steps, dt = 0.0, 0, 0.0
    while time < final_time:
        at_quad = torch.einsum("xyklv,ak,bl->xyabv", coeff, phi, phi)
        h_vals = at_quad[..., 0].clamp_min(1e-12)
        u_vals = at_quad[..., 1] / h_vals
        v_vals = at_quad[..., 2] / h_vals
        c_vals = torch.sqrt(GRAVITY * h_vals)
        max_speed = float(torch.max(torch.sqrt(u_vals ** 2 + v_vals ** 2) + c_vals).item())
        dt = min(cfl * min(dx, dy) / ((2 * degree + 1) * max(max_speed, 1e-12)), final_time - time)
        if dt < 1e-15:
            break

        k1 = _rhs_2d(coeff, dx, dy, weights, phi, dphi, phi_left, phi_right,
                     zb_jump_x, zb_jump_y, periodic,
                     bed_modal, zb_slope_quad_x, zb_slope_quad_y)
        u1 = coeff + dt * k1
        u1 = _positivity_filter_2d(_limit(u1), check_points, h_min)

        k2 = _rhs_2d(u1, dx, dy, weights, phi, dphi, phi_left, phi_right,
                     zb_jump_x, zb_jump_y, periodic,
                     bed_modal, zb_slope_quad_x, zb_slope_quad_y)
        u2 = 0.75 * coeff + 0.25 * (u1 + dt * k2)
        u2 = _positivity_filter_2d(_limit(u2), check_points, h_min)

        k3 = _rhs_2d(u2, dx, dy, weights, phi, dphi, phi_left, phi_right,
                     zb_jump_x, zb_jump_y, periodic,
                     bed_modal, zb_slope_quad_x, zb_slope_quad_y)
        coeff = (1.0 / 3.0) * coeff + (2.0 / 3.0) * (u2 + dt * k3)
        coeff = _positivity_filter_2d(_limit(coeff), check_points, h_min)

        if not torch.isfinite(coeff).all():
            raise FloatingPointError("Nonphysical 2D SWE state")

        time += dt
        steps += 1

    zb_faces = {"x": (zb_lx, zb_rx), "y": (zb_ly, zb_ry)}
    metadata = {
        "nx": nx, "ny": ny, "degree": degree, "final_time": final_time,
        "steps": steps, **step_metadata(final_time, steps, dt),
        "domain": domain, "cfl": cfl,
        "limiter": limiter, "device": str(dev),
        "well_balanced": well_balanced,
        "bed_representation": ("nodal_lgl_polynomial" if well_balanced
                               else "cell_constant_slope"),
        "zb_modes": bed_modal,
        "time_reached": time,
        "dry_depth": DRY_DEPTH,
    }
    return x, y, coeff, zb_faces, metadata


# ============================================================================
# Hydrostatic well-balanced helpers (2-D)
# ============================================================================

def reconstruct_fields_2d(coeff: torch.Tensor, zb_modes=None) -> tuple:
    """Reconstruct h, u, v (and η when the bed is known) at cell centres.

    ``coeff`` is the ``(nx, ny, nmodes, nmodes, 3)`` modal array returned by
    :func:`solve_swe_2d`.  ``zb_modes`` is the ``(nx, ny, nmodes, nmodes)`` bed
    polynomial of the well-balanced path (``metadata["zb_modes"]``); its mode
    ``(0, 0)`` is the cell-average bed, which is the correct value to add to the
    cell-average depth.  Without it the free surface is not returned.

    Returns ``(h, u, v, eta)`` with ``eta`` equal to ``None`` when
    ``zb_modes`` is not given.
    """
    h = coeff[:, :, 0, 0, 0].clamp_min(0.0)
    hu = coeff[:, :, 0, 0, 1]
    hv = coeff[:, :, 0, 0, 2]
    wetted = h > DRY_DEPTH
    u = torch.where(wetted, hu / h.clamp_min(DRY_DEPTH), torch.zeros_like(h))
    v = torch.where(wetted, hv / h.clamp_min(DRY_DEPTH), torch.zeros_like(h))
    eta = None if zb_modes is None else h + zb_modes[:, :, 0, 0]
    return h, u, v, eta


def check_hydrostatic_equilibrium_2d(coeff: torch.Tensor, zb_modes=None,
                                     eta_expected: Optional[float] = None,
                                     tolerance: float = 1e-8) -> dict:
    """Check the 2-D lake-at-rest condition ``η = const``, ``u = v = 0``.

    Wet cells only: a dry cell keeps ``h = 0`` by convention, so its ``η`` is
    the bed level and must not be compared with the still-water level.
    """
    h, u, v, eta = reconstruct_fields_2d(coeff, zb_modes)
    wet = h > DRY_DEPTH
    if eta is None:
        eta = h
    if eta_expected is None:
        eta_expected = float(eta[wet].mean()) if bool(wet.any()) else 0.0
    eta_error = torch.where(wet, eta - eta_expected, torch.zeros_like(eta))
    eta_error_max = float(eta_error.abs().max())
    u_max = float(u.abs().max())
    v_max = float(v.abs().max())
    is_hydrostatic = (eta_error_max < tolerance and u_max < tolerance
                      and v_max < tolerance)
    message = (
        f"Hydrostatic equilibrium {'satisfied' if is_hydrostatic else 'NOT satisfied'}: "
        f"η error = {eta_error_max:.3e} m, |u| = {u_max:.3e} m/s, "
        f"|v| = {v_max:.3e} m/s (tol = {tolerance:.1e})"
    )
    return {
        "is_hydrostatic": is_hydrostatic,
        "eta_error_max": eta_error_max,
        "u_max": u_max,
        "v_max": v_max,
        "eta_expected": eta_expected,
        "message": message,
    }
