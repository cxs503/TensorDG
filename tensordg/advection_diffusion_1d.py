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
    ncells: int = 32,
    degree: int = 2,
    peclet: float = 10.0,
    speed: float = 1.0,
    penalty: float = 20.0,
    left_value: float = 0.0,
    right_value: float = 1.0,
    advection_scheme: str = "upwind",
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    *, source=0., reaction=0., total_flux=(None,None), mean_value=None,
):
    """Solve (1) on [0, 1] and return (cell_centres, modal_coefficients, metadata).

    Parameters
    ----------
    peclet : float
        Peclet magnitude ``|a| L / D`` with ``L = 1``; the diffusivity is
        ``D = abs(speed) / peclet``.
    source, reaction
        Solve (a u - D u')' + reaction*u = source. Each coefficient accepts a
        scalar, one value per cell, or a callable of quadrature coordinates
        of shape (ncells, nq). The reaction coefficient must be nonnegative.
    total_flux : (left, right)
        A finite scalar prescribes the rightward advective plus diffusive
        flux in place of that end's Dirichlet value. None uses Dirichlet data.
        With two prescribed fluxes and zero reaction, supply mean_value
        to select a unique solution; the flux difference must equal the
        integrated source. The unit interval has length one.
    advection_scheme : {"upwind", "downwind"}
        ``"downwind"`` exists only as an implementation negative control: it is
        *not* a consistent scheme and must fail the verification gates.
    """
    if ncells < 2 or degree not in (1, 2, 3):
        raise ValueError("ncells >= 2 and degree in [1, 2, 3] are required")
    if peclet <= 0.0 or speed == 0.0:
        raise ValueError("peclet > 0 and speed != 0 are required")
    if penalty <= 0.0:
        raise ValueError("penalty must be positive")
    if advection_scheme not in ("upwind", "downwind"):
        raise ValueError("advection_scheme must be 'upwind' or 'downwind'")

    dev = torch.device(device)
    h = 1.0 / ncells
    modes = degree + 1
    ndofs = ncells * modes
    centers = (torch.arange(ncells, device=dev, dtype=dtype) + 0.5) * h
    (
        nodes, weights, phi, stiffness, volume, phi_left, phi_right,
        dphi_left, dphi_right,
    ) = _operators(degree, dev, dtype)
    diffusivity = abs(speed) / peclet
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
    diffusion_volume = diffusivity * (2.0 / h) * stiffness
    advection_volume = -a * volume
    xq = centers[:, None] + (.5*h)*nodes
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
    needs_mean = west_flux is not None and east_flux is not None and not torch.any(sink>0)
    if needs_mean and (mean_value is None or not math.isfinite(mean_value)):
        raise ValueError("two total flux boundaries require a finite mean_value or nonzero reaction")
    if not needs_mean and mean_value is not None:
        raise ValueError("mean_value requires two prescribed fluxes and zero reaction")
    reaction_volume = (.5*h)*torch.einsum("cq,q,qi,qj->cij",sink,weights,phi,phi)
    integrated_source = (.5*h)*profile(source,"source")*weights
    if needs_mean:
        residual = integrated_source.sum() + west_flux - east_flux
        scale = integrated_source.abs().sum() + abs(west_flux) + abs(east_flux)
        if residual.abs() > 64*torch.finfo(dtype).eps*scale:
            raise ValueError("incompatible total flux difference and integrated source")
    rhs = (integrated_source @ phi).reshape(-1)
    for cell in range(ncells):
        idx = dofs(cell)
        add(idx, idx, diffusion_volume + advection_volume + reaction_volume[cell])

    # ---- interior diffusive faces: SIPG face block (as in tensordg.heat_1d) --
    jump = torch.cat((phi_right, -phi_left))
    average_derivative = 0.5 * diffusivity * (2.0 / h) * torch.cat((dphi_right, dphi_left))
    consistency = -torch.outer(jump, average_derivative)
    penalty_face = penalty * modes * modes / h
    diffusion_face = consistency + consistency.T + penalty_face * torch.outer(jump, jump)
    for cell in range(ncells - 1):
        idx = torch.cat((dofs(cell), dofs(cell + 1)))
        add(idx, idx, diffusion_face)

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
        derivative = outward * diffusivity * (2.0 / h) * derivative_endpoint
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
        "advection_scheme": advection_scheme, "h": h,
        "device": str(dev),
    }
    return centers, coeff, metadata


def advection_diffusion_error(
    centers: torch.Tensor, coeff: torch.Tensor, peclet: float,
    speed: float = 1.0, left_value: float = 0.0, right_value: float = 1.0,
) -> dict[str, float]:
    """Quadrature L2 error of the discrete solution against the exact layer."""
    degree = coeff.shape[1] - 1
    h = 1.0 / centers.numel()
    nodes_np, weights_np = np.polynomial.legendre.leggauss(max(2 * degree + 5, 8))
    nodes = torch.as_tensor(nodes_np, device=coeff.device, dtype=coeff.dtype)
    weights = torch.as_tensor(weights_np, device=coeff.device, dtype=coeff.dtype)
    phi = _legendre_values(nodes_np, degree, coeff.device, coeff.dtype)
    numerical = coeff @ phi.T
    x = centers[:, None] + 0.5 * h * nodes[None, :]
    exact = exact_solution(x, peclet, speed, left_value, right_value)
    error2 = 0.5 * h * torch.sum(weights[None, :] * (numerical - exact) ** 2)
    norm2 = 0.5 * h * torch.sum(weights[None, :] * exact ** 2)
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
    centers: torch.Tensor, coeff: torch.Tensor, samples_per_cell: int = 8,
):
    """Return ``(x, u_h)`` sampled at Gauss points inside every cell (1-D arrays)."""
    degree = coeff.shape[1] - 1
    nodes_np, _ = np.polynomial.legendre.leggauss(max(samples_per_cell, degree + 2))
    h = 1.0 / centers.numel()
    phi = _legendre_values(nodes_np, degree, coeff.device, coeff.dtype)
    x = (centers[:, None] + 0.5 * h * torch.as_tensor(
        nodes_np, device=coeff.device, dtype=coeff.dtype,
    )[None, :]).reshape(-1)
    values = (coeff @ phi.T).reshape(-1)
    return x, values


def face_fluxes(
    coeff: torch.Tensor, peclet: float, speed: float = 1.0,
) -> torch.Tensor:
    """Total flux ``a u - D u'`` evaluated at every cell's right face.

    For the exact solution every entry equals :func:`exact_flux`; the spread of
    the returned vector is therefore a direct measure of face-coupling
    consistency, independent of the L2 error norm.
    """
    degree = coeff.shape[1] - 1
    h = 1.0 / coeff.shape[0]
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
    return speed * value - diffusivity * derivative


def numerical_face_fluxes(coeff, peclet, speed=1., penalty=20., left_value=0., right_value=1.,
                          advection_scheme="upwind", total_flux=(None,None)):
    """Single-valued rightward DG total flux at all n+1 faces, including Nitsche terms."""
    if advection_scheme not in ("upwind","downwind"):
        raise ValueError("unknown advection scheme")
    ncells,modes = coeff.shape
    _,_,_,_,_,pl,pr,dl,dr = _operators(modes-1,coeff.device,coeff.dtype)
    west,east = coeff @ pl,coeff @ pr
    west_flux,east_flux = total_flux
    low = torch.cat((west.new_tensor([left_value if west_flux is None else 0.]),east))
    high = torch.cat((west,east.new_tensor([right_value if east_flux is None else 0.])))
    diff_scale = -2*ncells*abs(speed)/peclet
    west_diff,east_diff = diff_scale*(coeff @ dl),diff_scale*(coeff @ dr)
    diffusion = torch.cat((west_diff[:1],.5*(east_diff[:-1]+west_diff[1:]),east_diff[-1:]))
    advected = low if (speed>0)==(advection_scheme=="upwind") else high
    flux = speed*advected + diffusion + penalty*modes**2*ncells*(low-high)
    if west_flux is not None:
        flux[0] = west_flux
    if east_flux is not None:
        flux[-1] = east_flux
    return flux
