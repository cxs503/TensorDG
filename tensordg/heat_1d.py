"""One-dimensional heat equation verified with SIPG and Crank--Nicolson time stepping.

Three backends are supported via the ``backend`` parameter:
  * ``"dense"``        -- assemble dense torch matrices, solve with torch.linalg.solve
  * ``"sparse"``       -- assemble scipy CSR matrices, solve with scipy.sparse.linalg.splu
  * ``"matrix-free"``  -- no global assembly, solve with preconditioned CG
"""
from __future__ import annotations

import math

import numpy as np
import torch


def exact_temperature(x: torch.Tensor, final_time) -> torch.Tensor:
    time = torch.as_tensor(final_time, device=x.device, dtype=x.dtype)
    return torch.sin(math.pi * x) * torch.exp(-math.pi ** 2 * time)


def _operators(degree: int, device: torch.device, dtype: torch.dtype):
    nq = max(2 * degree + 4, 6)
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
            np.polynomial.legendre.legval(nodes_np, polynomial), device=device, dtype=dtype,
        ))
        deriv_values = (
            np.polynomial.legendre.legval(nodes_np, derivative)
            if len(derivative) else np.zeros_like(nodes_np)
        )
        derivatives.append(torch.as_tensor(deriv_values, device=device, dtype=dtype))
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
    mass = torch.einsum("q,qk,qi->ki", weights, phi, phi)
    stiffness = torch.einsum("q,qk,qi->ki", weights, dphi, dphi)
    tensor = lambda data: torch.as_tensor(data, device=device, dtype=dtype)
    return (
        nodes, weights, phi, mass, stiffness, tensor(left), tensor(right),
        tensor(deriv_left), tensor(deriv_right),
    )


def _project_initial(centers, h, nodes, weights, phi, initial_temperature):
    """Quadrature L2 projection onto the cellwise Legendre basis."""
    x = centers[:, None] + .5*h*nodes
    values = torch.sin(math.pi*x) if initial_temperature is None else initial_temperature(x)
    values = torch.as_tensor(values, device=x.device, dtype=x.dtype).expand_as(x)
    if not torch.isfinite(values).all():
        raise ValueError("initial temperature must be finite at quadrature nodes")
    modes = torch.arange(phi.shape[1], dtype=x.dtype, device=x.device)
    return ((values*weights) @ phi) * (modes + .5)


def _source_load(centers, h, nodes, weights, phi, source):
    """Integrate a stationary physical source against the modal test functions."""
    x = centers[:, None] + .5*h*nodes
    values = source(x) if callable(source) else source
    values = torch.as_tensor(values, dtype=x.dtype, device=x.device).expand_as(x)
    if not torch.isfinite(values).all():
        raise ValueError("source must be finite at quadrature nodes")
    return (.5*h)*((values*weights) @ phi)


def _wall_pair(wall_temperatures, like):
    if wall_temperatures is None:
        return like.new_zeros(2)
    pair = torch.as_tensor(wall_temperatures, device=like.device, dtype=like.dtype)
    if pair.shape != (2,) or not torch.isfinite(pair).all():
        raise ValueError("wall_temperatures must contain two finite temperatures")
    return pair


def _stationary_load(centers, h, operators, source, penalty, wall_temperatures):
    nodes, weights, phi, _, _, left, right, dleft, dright = operators
    load = None if source is None else _source_load(centers,h,nodes,weights,phi,source)
    if wall_temperatures is not None:
        walls = _wall_pair(wall_temperatures, centers)
        if load is None:
            load = centers.new_zeros((centers.numel(), phi.shape[1]))
        strength = penalty*phi.shape[1]**2/h
        # The Nitsche load is g*(penalty*v - outward_normal*dv/dx).
        load[0] = load[0] + walls[0]*(strength*left + (2/h)*dleft)
        load[-1] = load[-1] + walls[1]*(strength*right - (2/h)*dright)
    return load


def _integrated_face_heat(coefficients, penalty, walls, duration):
    integrated_walls = duration*_wall_pair(walls, coefficients)
    return sipg_heat_face_fluxes(coefficients, penalty, wall_temperatures=integrated_walls).tolist()


def _add_block(matrix, indices, block):
    matrix[indices[:, None], indices[None, :]] += block


def assemble_sipg_1d(
    ncells: int = 16,
    degree: int = 2,
    penalty: float = 20.0,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    *, initial_temperature=None, source=None, wall_temperatures=None,
):
    """Assemble the SIPG mass/diffusion matrices for ``u_t = u_xx`` on (0,1).

    source_load includes prescribed volumetric and Dirichlet terms.
    The same operator is used by the Crank--Nicolson driver
    :func:`solve_heat_1d`, the matrix-exponential driver
    :func:`solve_heat_1d_exact_time` and any external study of the semi-discrete
    spectrum, so the discrete operator has a single source of truth.

    Returns a dict with ``centers``, ``h``, ``dofs``, ``mass_matrix`` (SPD),
    ``diffusion`` (the SPD SIPG stiffness ``K`` of ``M du/dt = -K u``, with face
    penalty ``penalty * (p+1)**2 / h`` on every interface and boundary face), and
    ``initial_coefficients`` (L2 projection of the supplied profile, default ``sin(pi*x)``).
    """
    if ncells < 2 or degree < 1 or penalty <= 0.0:
        raise ValueError("ncells >= 2, degree >= 1, and positive penalty are required")
    dev = torch.device(device)
    h = 1.0 / ncells
    centers = (torch.arange(ncells, device=dev, dtype=dtype) + 0.5) * h
    operators = _operators(degree, dev, dtype)
    nodes, weights, phi, mass, stiffness, phi_left, phi_right, dphi_left, dphi_right = operators
    modes = degree + 1
    ndofs = ncells * modes
    diffusion = torch.zeros((ndofs, ndofs), device=dev, dtype=dtype)
    mass_matrix = torch.zeros_like(diffusion)
    local = torch.arange(modes, device=dev)

    def dofs(cell):
        return cell * modes + local

    volume_diffusion = (2.0 / h) * stiffness
    volume_mass = 0.5 * h * mass
    for cell in range(ncells):
        indices = dofs(cell)
        _add_block(diffusion, indices, volume_diffusion)
        _add_block(mass_matrix, indices, volume_mass)

    jump = torch.cat((phi_right, -phi_left))
    average_derivative = 0.5 * (2.0 / h) * torch.cat((dphi_right, dphi_left))
    consistency = -torch.outer(jump, average_derivative)
    face = consistency + consistency.T + penalty * modes * modes / h * torch.outer(jump, jump)
    for cell in range(ncells - 1):
        indices = torch.cat((dofs(cell), dofs(cell + 1)))
        _add_block(diffusion, indices, face)

    for cell, endpoint, derivative_endpoint, outward in (
        (0, phi_left, dphi_left, -1.0),
        (ncells - 1, phi_right, dphi_right, 1.0),
    ):
        derivative = outward * (2.0 / h) * derivative_endpoint
        consistency = -torch.outer(endpoint, derivative)
        block = consistency + consistency.T + penalty * modes * modes / h * torch.outer(
            endpoint, endpoint,
        )
        _add_block(diffusion, dofs(cell), block)

    coeff = _project_initial(centers, h, nodes, weights, phi, initial_temperature)
    return {
        "ncells": ncells,
        "degree": degree,
        "h": h,
        "dofs": ndofs,
        "centers": centers,
        "mass_matrix": mass_matrix,
        "diffusion": diffusion,
        "initial_coefficients": coeff,
        "source_load": _stationary_load(centers,h,operators,source,penalty,wall_temperatures),
        "penalty": penalty,
    }


from .registry import capability
@capability(
    name="heat-1d",
    category="solver",
    maturity="planned",
    evidence={"feature_ledger": "CASE-DG-ADVECT-ORDER"},
    description="1D heat equation DG",
    tags=['heat', '1d', 'parabolic']
)
def solve_heat_1d(
    ncells: int = 16,
    degree: int = 2,
    final_time: float = 0.1,
    time_steps: int | None = None,
    penalty: float = 20.0,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    backend: str = "dense",
    *, integrate_face_fluxes: bool = False, initial_temperature=None, source=None, wall_temperatures=None,
):
    """Solve the forced heat equation with fixed Dirichlet data and a projected initial profile.

    Crank--Nicolson is used for the global method-of-lines system. If
    ``time_steps`` is omitted, dt is approximately ``0.02*cell_width``.

    Parameters
    ----------
    wall_temperatures : optional pair
        Constant left and right temperatures, default (0, 0).
    source : optional scalar or callable of x
        Stationary volumetric source for u_t = u_xx + source on the unit slab.
    initial_temperature : optional callable
        Evaluated on cell quadrature points and projected onto the modal basis.
        The default profile is sin(pi*x).
    integrate_face_fluxes : bool
        Include the time-integrated flux toward +x at all faces in metadata
        under "integrated_face_fluxes", using Crank–Nicolson stage values.
    backend : str
        - ``"dense"``: assemble dense torch matrices (default, backward compatible)
        - ``"sparse"``: assemble scipy CSR matrices, solve with splu
        - ``"matrix-free"``: no global assembly, solve with PCG per step
    """
    if backend not in ("dense", "sparse", "matrix-free"):
        raise ValueError("backend must be 'dense', 'sparse', or 'matrix-free'")

    if backend == "sparse":
        from tensordg.sipg_sparse import solve_heat_1d_sparse
        return solve_heat_1d_sparse(
            ncells, degree, final_time, time_steps, penalty, device, dtype,
            integrate_face_fluxes=integrate_face_fluxes, initial_temperature=initial_temperature, source=source, wall_temperatures=wall_temperatures,
        )
    elif backend == "matrix-free":
        from tensordg.sipg_matrixfree import solve_heat_1d_matrixfree
        return solve_heat_1d_matrixfree(
            ncells, degree, final_time, time_steps, penalty, device, dtype,
            integrate_face_fluxes=integrate_face_fluxes, initial_temperature=initial_temperature, source=source, wall_temperatures=wall_temperatures,
        )

    # backend == "dense": original implementation
    if ncells < 2 or degree < 1 or penalty <= 0.0:
        raise ValueError("ncells >= 2, degree >= 1, and positive penalty are required")
    if final_time < 0.0 or (time_steps is not None and time_steps < 1):
        raise ValueError("final_time must be >= 0 and time_steps must be positive")
    dev = torch.device(device)
    assembly = assemble_sipg_1d(
        ncells=ncells, degree=degree, penalty=penalty, device=device, dtype=dtype, initial_temperature=initial_temperature, source=source, wall_temperatures=wall_temperatures,
    )
    h = assembly["h"]
    modes = degree + 1
    centers = assembly["centers"]
    mass_matrix = assembly["mass_matrix"]
    diffusion = assembly["diffusion"]
    coeff = assembly["initial_coefficients"]
    ndofs = assembly["dofs"]
    steps = (
        max(1, math.ceil(final_time / (0.02 * h)))
        if time_steps is None else time_steps
    ) if final_time > 0.0 else 0
    dt = final_time / steps if steps else 0.0
    state = coeff.reshape(-1)
    integrated = torch.zeros_like(state) if integrate_face_fluxes else None
    if steps:
        load = assembly["source_load"]
        increment = 0. if load is None else dt*load.reshape(-1)
        system = mass_matrix + 0.5 * dt * diffusion
        explicit = mass_matrix - 0.5 * dt * diffusion
        for _ in range(steps):
            previous = state
            state = torch.linalg.solve(system, explicit @ state + increment)
            if integrated is not None:
                integrated += (.5 * dt) * (previous + state)
    coeff = state.reshape(ncells, modes)
    metadata = {
        "ncells": ncells, "degree": degree, "final_time": final_time,
        "steps": steps, "dt": dt, "penalty": penalty, "dofs": ndofs,
        "diffusion_symmetry_error": float(torch.max(torch.abs(diffusion - diffusion.T)).item()),
        "device": str(dev),
    }
    if integrated is not None:
        integrated_coeff = torch.as_tensor(integrated, dtype=dtype, device=dev).reshape(ncells, modes)
        metadata["integrated_face_fluxes"] = _integrated_face_heat(integrated_coeff, penalty, wall_temperatures, final_time)
    return centers, coeff, metadata


def solve_heat_1d_exact_time(
    ncells: int = 16,
    degree: int = 2,
    final_time: float = 0.1,
    penalty: float = 20.0,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    *, initial_temperature=None, source=None, wall_temperatures=None, integrate_face_fluxes: bool = False,
):
    """SIPG heat solution with the semi-discrete system integrated *exactly* in time.

    A stationary source is integrated by augmenting the state with a constant.
    For zero source the semi-discrete system is linear and autonomous, ``M du/dt = -K u``, so the
    solution is ``u(t) = exp(-t M^{-1} K) u(0)``.  Evaluating the matrix
    exponential (SciPy's Pade implementation) removes the temporal error of
    Crank--Nicolson, which for high polynomial degrees is otherwise *larger*
    than the spatial error and masks the ``h**(p+1)`` rate.  This makes the
    driver a pure measurement of the **spatial** order for any ``p``.

    When requested, integrated_face_fluxes gives cumulative rightward heat
    at each face from the time integral of the semi-discrete balance.

    Returns ``(centers, coefficients, metadata)`` like :func:`solve_heat_1d`.
    """
    if final_time < 0.0:
        raise ValueError("final_time must be >= 0")
    from scipy.linalg import expm

    system = assemble_sipg_1d(
        ncells=ncells, degree=degree, penalty=penalty, device=device, dtype=dtype, initial_temperature=initial_temperature, source=source, wall_temperatures=wall_temperatures,
    )
    mass = system["mass_matrix"].detach().cpu().numpy()
    diffusion = system["diffusion"].detach().cpu().numpy()
    initial = system["initial_coefficients"].reshape(-1).detach().cpu().numpy()
    generator = np.linalg.solve(mass, diffusion)
    # ``diffusion`` holds the SIPG stiffness K with M du/dt = -K u (the
    # Crank--Nicolson sweep below solves (M + dt/2 K) u^{n+1} = (M - dt/2 K) u^n),
    # so the semi-discrete generator is -M^{-1} K and has negative real spectrum.
    eigenvalues = np.linalg.eigvals(generator)
    if eigenvalues.real.min() <= 0.0:
        raise RuntimeError("the SIPG stiffness is not positive definite")
    state = initial
    if final_time > 0.:
        if system["source_load"] is None:
            state = expm(-final_time * generator) @ initial
        else:
            load = system["source_load"].detach().cpu().numpy().reshape(-1)
            augmented = np.zeros((len(initial)+1, len(initial)+1), dtype=mass.dtype)
            augmented[:-1, :-1] = -generator
            augmented[:-1, -1] = np.linalg.solve(mass, load)
            state = (expm(final_time*augmented) @ np.r_[initial, 1.])[:-1]
    coeff = torch.as_tensor(
        state.reshape(ncells, degree + 1),
        device=torch.device(device), dtype=dtype,
    )
    metadata = {
        "ncells": ncells,
        "degree": degree,
        "final_time": final_time,
        "penalty": penalty,
        "dofs": system["dofs"],
        "time_integration": "matrix-exponential",
        "mass_symmetry_error": float(np.max(np.abs(mass - mass.T))),
        "diffusion_symmetry_error": float(np.max(np.abs(diffusion - diffusion.T))),
        "stiffness_eigenvalue_min": float(eigenvalues.real.min()),
        "stiffness_eigenvalue_max": float(eigenvalues.real.max()),
        "time_step_limit": float(2.0 / eigenvalues.real.max()),
    }
    if integrate_face_fluxes:
        # Integrate M du/dt = f-Ku before recovering the numerical face flux.
        rhs = mass @ (initial-state)
        if system["source_load"] is not None:
            rhs += final_time*system["source_load"].detach().cpu().numpy().reshape(-1)
        integrated = np.linalg.solve(diffusion, rhs).reshape(ncells, degree+1)
        modes = torch.as_tensor(integrated, device=coeff.device, dtype=coeff.dtype)
        metadata["integrated_face_fluxes"] = _integrated_face_heat(modes, penalty, wall_temperatures, final_time)
    return system["centers"], coeff, metadata


def heat_error(
    centers: torch.Tensor, coeff: torch.Tensor, final_time: float, *, reference=exact_temperature,
) -> dict[str, float]:
    """Quadrature error against reference(x, time), defaulting to the first sine mode."""
    degree = coeff.shape[1] - 1
    h = 1.0 / centers.numel()
    nodes_np, weights_np = np.polynomial.legendre.leggauss(max(2 * degree + 5, 8))
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
    numerical = coeff @ phi.T
    x = centers[:, None] + 0.5 * h * nodes[None, :]
    exact = reference(x, final_time)
    error2 = 0.5 * h * torch.sum(weights[None, :] * (numerical - exact) ** 2)
    norm2 = 0.5 * h * torch.sum(weights[None, :] * exact ** 2)
    absolute = float(torch.sqrt(error2).item())
    norm = float(torch.sqrt(norm2).item())
    return {"absolute_l2": absolute, "exact_l2": norm, "relative_l2": absolute / norm}


def sipg_heat_face_fluxes(coefficients: torch.Tensor, penalty: float = 20., *, wall_temperatures=None) -> torch.Tensor:
    """Return conductive flux toward +x at all faces of the uniform unit slab.

    Uses the modal coefficients returned by solve_heat_1d, unit diffusivity,
    the prescribed left and right wall temperatures (default zero) and the solver's SIPG penalty parameter.
    For cell i the inward net power is flux[i] - flux[i+1].
    """
    if coefficients.ndim != 2 or min(coefficients.shape) < 2:
        raise ValueError("expected at least two cells and two Legendre modes")
    if not math.isfinite(penalty) or penalty <= 0:
        raise ValueError("penalty must be finite and positive")
    cells, modes = coefficients.shape
    *_, left, right, dleft, dright = _operators(modes-1, coefficients.device, coefficients.dtype)
    trace_l, trace_r = coefficients @ left, coefficients @ right
    grad_l, grad_r = (2*cells)*(coefficients @ dleft), (2*cells)*(coefficients @ dright)
    strength = penalty*modes**2*cells
    interior = -.5*(grad_r[:-1]+grad_l[1:]) + strength*(trace_r[:-1]-trace_l[1:])
    walls = _wall_pair(wall_temperatures, coefficients)
    return torch.cat((-grad_l[:1]-strength*(trace_l[:1]-walls[0]), interior,
                      -grad_r[-1:]+strength*(trace_r[-1:]-walls[1])))
