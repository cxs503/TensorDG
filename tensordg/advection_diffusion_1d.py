"""Steady one-dimensional advection--diffusion DG benchmark with an exact layer solution.

The merged TensorDG operator stack has two separately verified halves -- upwind DG
advection (:mod:`tensordg.advection_1d`) and symmetric interior penalty (SIPG)
diffusion (:mod:`tensordg.heat_1d`, :mod:`tensordg.sipg_sparse`,
:mod:`tensordg.sipg_matrixfree`) -- but the two were never exercised *together*,
and the 1-D SIPG solvers only ever ran with **homogeneous** Dirichlet data.  This
module closes that gap with the classical exponential-boundary-layer problem

    a u'(x) - D u''(x) = 0,   u(0) = g_L,  u(1) = g_R,   a != 0, D > 0,      (1)

whose exact solution is

    u(x) = g_L + (g_R - g_L) * (exp(Pe x) - 1) / (exp(Pe) - 1),
    Pe  = a L / D (signed in this formula)                                                          (2)

and whose exact *total flux* is constant in x,

    J = a u - D u' = a * (g_L - (g_R - g_L) / (exp(Pe) - 1)).                      (3)

Equation (3) is the discrete-consistency meter used by the tests: for a
consistent DG discretisation the fluxes recovered on every cell face agree with
each other (and with (3)) to discretisation accuracy, so it detects broken face
coupling that an L2 error norm can hide.

Discretisation
--------------
Modal Legendre basis (identical conventions to :mod:`tensordg.advection_1d` and
:mod:`tensordg.heat_1d`), upwind flux for the advective face term, SIPG for the
diffusive term with the penalty ``penalty * (degree+1)**2 / h`` used by
``heat_1d``, and inhomogeneous Dirichlet data imposed through the Nitsche-type
boundary terms.  Everything is assembled as one dense global system, so the
solver stays O(N^3) with N = ncells*(degree+1) and is meant for verification on
small grids, not for production runs.
"""
from __future__ import annotations

import math

import numpy as np
import torch


def exact_solution(
    x: torch.Tensor, peclet: float, speed: float = 1.0,
    left_value: float = 0.0, right_value: float = 1.0,
) -> torch.Tensor:
    """Dirichlet layer profile for Pe = |a|/D on the unit interval."""
    fraction = torch.expm1(-peclet * x) / math.expm1(-peclet)
    if speed > 0:
        fraction = fraction * torch.exp(peclet * (x - 1.0))
    return left_value + (right_value - left_value) * fraction


def exact_flux(
    peclet: float, speed: float = 1.0, left_value: float = 0.0, right_value: float = 1.0,
) -> float:
    """Exact constant total flux with the sign of the physical x direction."""
    inflow, outflow = (left_value, right_value) if speed > 0 else (right_value, left_value)
    correction = (outflow - inflow) * math.exp(-peclet) / -math.expm1(-peclet)
    return speed * (inflow - correction)


def _operators(degree: int, device: torch.device, dtype: torch.dtype):
    nq = max(2 * degree + 4, 6)
    nodes_np, weights_np = np.polynomial.legendre.leggauss(nq)
    nodes = torch.as_tensor(nodes_np, device=device, dtype=dtype)
    weights = torch.as_tensor(weights_np, device=device, dtype=dtype)
    values, derivatives, left, right, deriv_left, deriv_right = [], [], [], [], [], []
    for k in range(degree + 1):
        polynomial = np.zeros(k + 1)
        polynomial[-1] = 1.0
        derivative = np.polynomial.legendre.legder(polynomial)
        values.append(torch.as_tensor(
            np.polynomial.legendre.legval(nodes_np, polynomial), device=device, dtype=dtype,
        ))
        derivatives.append(torch.as_tensor(
            np.polynomial.legendre.legval(nodes_np, derivative)
            if len(derivative) else np.zeros_like(nodes_np),
            device=device, dtype=dtype,
        ))
        left.append((-1.0) ** k)
        right.append(1.0)
        deriv_left.append(
            float(np.polynomial.legendre.legval(-1.0, derivative)) if len(derivative) else 0.0,
        )
        deriv_right.append(
            float(np.polynomial.legendre.legval(1.0, derivative)) if len(derivative) else 0.0,
        )
    phi = torch.stack(values, dim=1)
    dphi = torch.stack(derivatives, dim=1)
    stiffness = torch.einsum("q,qk,qi->ki", weights, dphi, dphi)
    # volume[i, j] = int_{-1}^{1} P_j dP_i/dxi dxi  (weak advective volume term)
    volume = torch.einsum("q,qj,qi->ij", weights, phi, dphi)
    tensor = lambda data: torch.as_tensor(data, device=device, dtype=dtype)
    return (
        nodes, weights, phi, stiffness, volume, tensor(left), tensor(right),
        tensor(deriv_left), tensor(deriv_right),
    )


def _mesh(ncells, cell_edges, device, dtype):
    if cell_edges is None:
        n = 32 if ncells is None else ncells
        if n < 2:
            raise ValueError("at least two cells are required")
        widths = torch.full((n,), 1./n, device=device, dtype=dtype)
        centers = (torch.arange(n, device=device, dtype=dtype)+.5)/n
        return centers, widths
    edges = torch.as_tensor(cell_edges, dtype=dtype, device=device)
    if edges.ndim != 1 or len(edges)<3 or (ncells is not None and len(edges)!=ncells+1):
        raise ValueError("cell_edges must contain one endpoint per face of at least two cells")
    widths = edges.diff()
    if not torch.isfinite(edges).all() or edges[0]!=0 or edges[-1]!=1 or torch.any(widths<=0):
        raise ValueError("cell_edges must increase strictly from 0 to 1")
    return edges[:-1]+.5*widths, widths


def _face_widths(widths):
    return torch.cat((widths[:1],torch.minimum(widths[:-1],widths[1:]),widths[-1:]))


def _material_ratios(value, widths):
    raw = torch.as_tensor(value,device=widths.device)
    if raw.is_complex() or raw.ndim>0 and raw.shape!=widths.shape:
        raise ValueError("diffusivity_ratio must be real and scalar or one value per cell")
    ratio = torch.as_tensor(value,device=widths.device,dtype=widths.dtype).expand_as(widths)
    if not torch.isfinite(ratio).all() or torch.any(ratio<=0):
        raise ValueError("diffusivity_ratio must be finite and positive")
    # Opposite-material weighting of the two physical diffusive traces gives
    # the harmonic coefficient multiplying their mean unscaled gradient.
    harmonic = 2/(ratio[:-1].reciprocal()+ratio[1:].reciprocal())
    return ratio,torch.cat((ratio[:1],harmonic,ratio[-1:]))


def _interface_profile(value, widths, name):
    raw = torch.as_tensor(value,device=widths.device)
    if raw.is_complex() or (raw.ndim and raw.shape!=widths[:-1].shape):
        raise ValueError(f"{name} must be real and scalar or one value per internal face")
    data = torch.as_tensor(value,device=widths.device,dtype=widths.dtype)
    if not torch.isfinite(data).all():
        raise ValueError(f"{name} must be finite")
    return data.expand_as(widths[:-1])


from .registry import capability
@capability(
    name="advection-diffusion-1d",
    category="solver",
    maturity="planned",
    evidence={"feature_ledger": "CASE-DG-ADVECT-ORDER"},
    description="1D advection-diffusion DG (Peclet sweep)",
    tags=['advection-diffusion', '1d']
)
def solve_advection_diffusion_1d(
    ncells: int | None = None,
    degree: int = 2,
    peclet: float = 10.0,
    speed: float = 1.0,
    penalty: float = 20.0,
    left_value: float = 0.0,
    right_value: float = 1.0,
    advection_scheme: str = "upwind",
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    *, source=0., reaction=0., total_flux=(None,None), mean_value=None, cell_edges=None, diffusivity_ratio=1., interface_source=0., interface_reaction=0.,
):
    """Solve (1) on [0, 1] and return (cell_centres, modal_coefficients, metadata).

    Parameters
    ----------
    peclet : float
        Peclet magnitude ``|a| L / D`` with ``L = 1``; the diffusivity is
        ``D = abs(speed) / peclet``.
    diffusivity_ratio : scalar or cell vector
        Multiply the reference diffusivity |speed|/peclet by these positive
        material ratios. Interfaces must coincide with mesh faces.
    interface_source : scalar or internal-face vector
        Point sources on mesh faces give J_right - J_left = interface_source.
        Positive values supply transported quantity per unit time.
    interface_reaction : nonnegative scalar or internal-face vector
        A first-order sink modifies the point source to S - k*u_interface.
        The concentration uses the same weighted test trace as the point load.
    cell_edges : optional vector
        Strictly increasing mesh edges spanning [0,1]. Omit ncells to infer
        it from the edges. The default mesh has 32 equal cells. Pass the same
        edges to sampling, error and flux diagnostics on a nonuniform grid.
    source, reaction
        Solve (a u - D u')' + reaction*u = source. Each coefficient accepts a
        scalar, one value per cell, or a callable of quadrature coordinates
        of shape (ncells, nq). The reaction coefficient must be nonnegative.
    total_flux : (left, right)
        A finite scalar prescribes the rightward advective plus diffusive
        flux in place of that end's Dirichlet value. None uses Dirichlet data.
        With two prescribed fluxes and zero volume and interface reaction, supply mean_value
        to select a unique solution; the flux difference must equal the
        integrated volume source plus all interface sources. The unit interval has length one.
    advection_scheme : {"upwind", "downwind"}
        ``"downwind"`` exists only as an implementation negative control: it is
        *not* a consistent scheme and must fail the verification gates.
    """
    if degree not in (1, 2, 3):
        raise ValueError("ncells >= 2 and degree in [1, 2, 3] are required")
    if peclet <= 0.0 or speed == 0.0:
        raise ValueError("peclet > 0 and speed != 0 are required")
    if penalty <= 0.0:
        raise ValueError("penalty must be positive")
    if advection_scheme not in ("upwind", "downwind"):
        raise ValueError("advection_scheme must be 'upwind' or 'downwind'")

    dev = torch.device(device)
    centers,h = _mesh(ncells,cell_edges,dev,dtype)
    ncells = len(h)
    modes = degree + 1
    ndofs = ncells * modes
    (
        nodes, weights, phi, stiffness, volume, phi_left, phi_right,
        dphi_left, dphi_right,
    ) = _operators(degree, dev, dtype)
    diffusivity = abs(speed) / peclet
    ratio,face_ratio = _material_ratios(diffusivity_ratio,h)
    local_diffusivity = diffusivity*ratio
    a = float(speed)

    system = torch.zeros((ndofs, ndofs), device=dev, dtype=dtype)
    local = torch.arange(modes, device=dev)
    dofs = lambda cell: cell * modes + local
    def add(rows, cols, block):
        system[rows[:, None], cols[None, :]] += block

    # ---- volume terms: SIPG diffusion + weak advection ----------------------
    # Steady residual convention:  B_diff(u, v) - A_adv(u, v) = L_diff(v), where
    # B_diff is the SIPG form of -D u'' (same matrix as tensordg.heat_1d) and
    # A_adv(u, v_i) = a * int_K u v_i' - a * f_hat_R v_i(1) + a * f_hat_L v_i(-1).
    # The two blocks therefore carry *opposite* signs.
    diffusion_volume = (2*local_diffusivity/h)[:,None,None] * stiffness
    advection_volume = -a * volume
    xq = centers[:, None] + (.5*h[:,None])*nodes
    def profile(value, name):
        sampled = value(xq) if callable(value) else value
        data = torch.as_tensor(sampled, device=dev)
        if data.is_complex():
            raise ValueError(f"{name} must be real")
        data = torch.as_tensor(sampled, device=dev, dtype=dtype)
        if data.shape == (ncells,):
            data = data[:, None]
        if data.ndim > 0 and data.shape not in ((ncells, 1), xq.shape):
            raise ValueError(f"{name} must be scalar, cellwise, or sampled at quadrature points")
        if not torch.isfinite(data).all():
            raise ValueError(f"{name} must be finite")
        return data.expand_as(xq)
    west_flux,east_flux = total_flux
    for value in (west_flux,east_flux):
        if value is not None and not math.isfinite(value):
            raise ValueError("boundary total flux must be finite")
    sink = profile(reaction,"reaction")
    if torch.any(sink < 0):
        raise ValueError("reaction must be nonnegative")
    uptake = _interface_profile(interface_reaction,h,"interface_reaction")
    if torch.any(uptake<0):
        raise ValueError("interface_reaction must be nonnegative")
    needs_mean = west_flux is not None and east_flux is not None and not torch.any(sink>0) and not torch.any(uptake>0)
    if needs_mean and (mean_value is None or not math.isfinite(mean_value)):
        raise ValueError("two total flux boundaries require a finite mean_value or nonzero reaction")
    if not needs_mean and mean_value is not None:
        raise ValueError("mean_value requires two prescribed fluxes and zero reaction")
    reaction_volume = (.5*h[:,None,None])*torch.einsum("cq,q,qi,qj->cij",sink,weights,phi,phi)
    integrated_source = (.5*h[:,None])*profile(source,"source")*weights
    sheet = _interface_profile(interface_source,h,"interface_source")
    if needs_mean:
        residual = sheet.sum() + integrated_source.sum() + west_flux - east_flux
        scale = sheet.abs().sum() + integrated_source.abs().sum() + abs(west_flux) + abs(east_flux)
        if residual.abs() > 64*torch.finfo(dtype).eps*scale:
            raise ValueError("incompatible total flux difference and integrated source")
    # A flux jump couples to the complementary average of the test traces:
    # J_minus = J_hat - w*S, J_plus = J_hat + (1-w)*S.
    w = ratio[:-1]/(ratio[:-1]+ratio[1:])
    cell_load = integrated_source @ phi
    cell_load[:-1] += (w*sheet)[:,None]*phi_right
    cell_load[1:] += ((1-w)*sheet)[:,None]*phi_left
    rhs = cell_load.reshape(-1)
    for cell in range(ncells):
        idx = dofs(cell)
        add(idx, idx, diffusion_volume[cell] + advection_volume + reaction_volume[cell])

    # ---- interior diffusive faces: SIPG face block (as in tensordg.heat_1d) --
    jump = torch.cat((phi_right, -phi_left))
    penalties = penalty*modes*modes*face_ratio/_face_widths(h)
    jump_matrix = torch.outer(jump,jump)
    for cell in range(ncells - 1):
        average_derivative = diffusivity*face_ratio[cell+1]*torch.cat((dphi_right/h[cell],dphi_left/h[cell+1]))
        consistency = -torch.outer(jump, average_derivative)
        diffusion_face = consistency + consistency.T + penalties[cell+1]*jump_matrix
        idx = torch.cat((dofs(cell), dofs(cell + 1)))
        trace = torch.cat((w[cell]*phi_right,(1-w[cell])*phi_left))
        add(idx, idx, diffusion_face + uptake[cell]*torch.outer(trace,trace))

    # ---- advective faces: one single-valued flux per face -------------------
    # The flux of face f is a * trace(source).  For the upwind scheme the source
    # is the element on the inflow side; the outflow boundary therefore uses the
    # interior trace while the inflow boundary uses the prescribed data.
    upwind = advection_scheme == "upwind"
    for face in range(ncells + 1):
        if (face==0 and west_flux is not None) or (face==ncells and east_flux is not None):
            continue
        left_cell = face - 1 if face > 0 else None
        right_cell = face if face < ncells else None
        src_is_left = (a > 0.0) == upwind
        source = left_cell if src_is_left else right_cell
        trace = phi_right if src_is_left else phi_left
        if source is None:
            boundary_value = left_value if face == 0 else right_value
            if left_cell is not None:
                rhs[dofs(left_cell)] += -a * boundary_value * phi_right
            if right_cell is not None:
                rhs[dofs(right_cell)] += a * boundary_value * phi_left
            continue
        if left_cell is not None:
            add(dofs(left_cell), dofs(source), a * torch.outer(phi_right, trace))
        if right_cell is not None:
            add(dofs(right_cell), dofs(source), -a * torch.outer(phi_left, trace))

    # ---- Dirichlet boundary terms for the diffusive operator (Nitsche) ------
    for cell, endpoint, derivative_endpoint, outward, value, prescribed_flux in (
        (0, phi_left, dphi_left, -1.0, left_value, west_flux),
        (ncells - 1, phi_right, dphi_right, 1.0, right_value, east_flux),
    ):
        idx = dofs(cell)
        if prescribed_flux is not None:
            rhs[idx] -= outward*prescribed_flux*endpoint
            continue
        derivative = outward * local_diffusivity[cell] * (2.0 / h[cell]) * derivative_endpoint
        penalty_face = penalties[0 if cell==0 else -1]
        consistency = -torch.outer(endpoint, derivative)
        block = consistency + consistency.T + penalty_face * torch.outer(endpoint, endpoint)
        add(idx, idx, block)
        # data terms of the Nitsche form: -int (D v' n) g + penalty * int g v
        rhs[idx] += (-derivative + penalty_face * endpoint) * value

    if needs_mean:
        # A bordered system retains each local balance and fixes the mean.
        # Its multiplier is a spatially uniform source, zero for compatible data.
        mean_row = rhs.new_zeros(ndofs)
        mean_row[::modes] = h
        constrained = system.new_zeros((ndofs+1,ndofs+1))
        constrained[:-1,:-1] = system
        constrained[-1,:-1] = mean_row
        constrained[:-1,-1] = mean_row
        state = torch.linalg.solve(constrained,torch.cat((rhs,rhs.new_tensor([mean_value]))))[:-1]
    else:
        state = torch.linalg.solve(system, rhs)
    coeff = state.reshape(ncells, modes)
    metadata = {
        "ncells": ncells, "degree": degree, "peclet": peclet, "speed": a,
        "diffusivity": diffusivity, "penalty": penalty, "dofs": ndofs,
        "advection_scheme": advection_scheme, "h": 1./ncells if cell_edges is None else None,
        "device": str(dev),
    }
    if torch.any(ratio!=1):
        metadata["diffusivity_ratio"] = ratio.tolist()
    if cell_edges is not None:
        metadata["cell_edges"] = torch.as_tensor(cell_edges,dtype=dtype).tolist()
    return centers, coeff, metadata


def advection_diffusion_error(
    centers: torch.Tensor, coeff: torch.Tensor, peclet: float,
    speed: float = 1.0, left_value: float = 0.0, right_value: float = 1.0, *, cell_edges=None,
) -> dict[str, float]:
    """Quadrature L2 error of the discrete solution against the exact layer."""
    degree = coeff.shape[1] - 1
    _, h = _mesh(centers.numel(),cell_edges,coeff.device,coeff.dtype)
    nodes_np, weights_np = np.polynomial.legendre.leggauss(max(2 * degree + 5, 8))
    nodes = torch.as_tensor(nodes_np, device=coeff.device, dtype=coeff.dtype)
    weights = torch.as_tensor(weights_np, device=coeff.device, dtype=coeff.dtype)
    phi = _legendre_values(nodes_np, degree, coeff.device, coeff.dtype)
    numerical = coeff @ phi.T
    x = centers[:, None] + 0.5 * h[:,None] * nodes[None, :]
    exact = exact_solution(x, peclet, speed, left_value, right_value)
    error2 = 0.5 * torch.sum(h[:,None] * weights[None, :] * (numerical - exact) ** 2)
    norm2 = 0.5 * torch.sum(h[:,None] * weights[None, :] * exact ** 2)
    absolute = float(torch.sqrt(error2).item())
    norm = float(torch.sqrt(norm2).item())
    return {"absolute_l2": absolute, "exact_l2": norm, "relative_l2": absolute / norm if norm else (0.0 if absolute == 0 else math.inf)}


def _legendre_values(nodes_np, degree: int, device, dtype) -> torch.Tensor:
    basis = []
    for k in range(degree + 1):
        polynomial = np.zeros(k + 1)
        polynomial[-1] = 1.0
        basis.append(torch.as_tensor(
            np.polynomial.legendre.legval(nodes_np, polynomial), device=device, dtype=dtype,
        ))
    return torch.stack(basis, dim=1)


def sample_solution(
    centers: torch.Tensor, coeff: torch.Tensor, samples_per_cell: int = 8, *, cell_edges=None,
):
    """Return ``(x, u_h)`` sampled at Gauss points inside every cell (1-D arrays)."""
    degree = coeff.shape[1] - 1
    nodes_np, _ = np.polynomial.legendre.leggauss(max(samples_per_cell, degree + 2))
    _, h = _mesh(centers.numel(),cell_edges,coeff.device,coeff.dtype)
    phi = _legendre_values(nodes_np, degree, coeff.device, coeff.dtype)
    x = (centers[:, None] + 0.5 * h[:,None] * torch.as_tensor(
        nodes_np, device=coeff.device, dtype=coeff.dtype,
    )[None, :]).reshape(-1)
    values = (coeff @ phi.T).reshape(-1)
    return x, values


def face_fluxes(
    coeff: torch.Tensor, peclet: float, speed: float = 1.0, *, cell_edges=None, diffusivity_ratio=1.,
) -> torch.Tensor:
    """Total flux ``a u - D u'`` evaluated at every cell's right face.

    For the exact solution every entry equals :func:`exact_flux`; the spread of
    the returned vector is therefore a direct measure of face-coupling
    consistency, independent of the L2 error norm.
    """
    degree = coeff.shape[1] - 1
    _, h = _mesh(coeff.shape[0],cell_edges,coeff.device,coeff.dtype)
    right_values, right_derivatives = [], []
    for k in range(degree + 1):
        polynomial = np.zeros(k + 1)
        polynomial[-1] = 1.0
        derivative = np.polynomial.legendre.legder(polynomial)
        right_values.append(1.0)
        right_derivatives.append(
            float(np.polynomial.legendre.legval(1.0, derivative)) if len(derivative) else 0.0
        )
    p_right = torch.as_tensor(right_values, device=coeff.device, dtype=coeff.dtype)
    dp_right = torch.as_tensor(right_derivatives, device=coeff.device, dtype=coeff.dtype)
    value = coeff @ p_right
    derivative = (2.0 / h) * (coeff @ dp_right)
    diffusivity = abs(speed) / peclet
    ratio,_ = _material_ratios(diffusivity_ratio,h)
    return speed * value - diffusivity * ratio * derivative


def numerical_face_fluxes(coeff, peclet, speed=1., penalty=20., left_value=0., right_value=1.,
                          advection_scheme="upwind", total_flux=(None,None), *, cell_edges=None, diffusivity_ratio=1.,
                          interface_source=0., interface_reaction=0., return_cell_fluxes=False):
    """Rightward DG flux at n+1 faces, with the weighted flux at internal faces.

    With return_cell_fluxes=True return (west, east), each of length ncells,
    including the source-induced jump across each internal face.
    """
    if advection_scheme not in ("upwind","downwind"):
        raise ValueError("unknown advection scheme")
    ncells,modes = coeff.shape
    _,_,_,_,_,pl,pr,dl,dr = _operators(modes-1,coeff.device,coeff.dtype)
    west,east = coeff @ pl,coeff @ pr
    west_flux,east_flux = total_flux
    low = torch.cat((west.new_tensor([left_value if west_flux is None else 0.]),east))
    high = torch.cat((west,east.new_tensor([right_value if east_flux is None else 0.])))
    _,widths = _mesh(ncells,cell_edges,coeff.device,coeff.dtype)
    ratio,face_ratio = _material_ratios(diffusivity_ratio,widths)
    sheet = _interface_profile(interface_source,widths,"interface_source")
    uptake = _interface_profile(interface_reaction,widths,"interface_reaction")
    if torch.any(uptake<0):
        raise ValueError("interface_reaction must be nonnegative")
    diff_scale = -2*abs(speed)/(peclet*widths)
    west_diff,east_diff = diff_scale*(coeff @ dl),diff_scale*(coeff @ dr)
    diffusion = torch.cat((west_diff[:1],.5*(east_diff[:-1]+west_diff[1:]),east_diff[-1:]))
    advected = low if (speed>0)==(advection_scheme=="upwind") else high
    flux = speed*advected + face_ratio*(diffusion + penalty*modes**2*(low-high)/_face_widths(widths))
    if west_flux is not None:
        flux[0] = west_flux
    if east_flux is not None:
        flux[-1] = east_flux
    if return_cell_fluxes:
        w = ratio[:-1]/(ratio[:-1]+ratio[1:])
        sheet = sheet-uptake*(w*east[:-1]+(1-w)*west[1:])
        west,east = flux[:-1].clone(),flux[1:].clone()
        west[1:] += (1-w)*sheet
        east[:-1] -= w*sheet
        return west,east
    return flux
