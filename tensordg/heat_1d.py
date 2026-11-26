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
    """Project a physical profile, or resume from its cellwise Legendre coefficients."""
    if isinstance(initial_temperature, torch.Tensor):
        if initial_temperature.shape != (centers.numel(), phi.shape[1]):
            raise ValueError("initial modal coefficients must have shape (ncells, degree+1)")
        if initial_temperature.is_complex() or not torch.isfinite(initial_temperature).all():
            raise ValueError("initial modal coefficients must be finite and real")
        return initial_temperature.to(centers)
    x = centers[:, None] + .5*h*nodes
    values = torch.sin(math.pi*x) if initial_temperature is None else initial_temperature(x)
    values = torch.as_tensor(values, device=x.device, dtype=x.dtype).expand_as(x)
    if not torch.isfinite(values).all():
        raise ValueError("initial temperature must be finite at quadrature nodes")
    modes = torch.arange(phi.shape[1], dtype=x.dtype, device=x.device)
    return ((values*weights) @ phi.to(values.dtype)) * (modes + .5)


def _source_load(centers, h, nodes, weights, phi, source, *, dtype=None):
    """Integrate a stationary physical source against the modal test functions."""
    x = centers[:, None] + .5*h*nodes
    values = source(x) if callable(source) else source
    values = torch.as_tensor(values, dtype=dtype or x.dtype, device=x.device).expand_as(x)
    if not torch.isfinite(values).all():
        raise ValueError("source must be finite at quadrature nodes")
    return (.5*h)*((values*weights) @ phi.to(values.dtype))


def _wall_pair(wall_temperatures, like):
    if wall_temperatures is None:
        return like.new_zeros(2)
    pair = torch.as_tensor(wall_temperatures, device=like.device, dtype=like.dtype)
    if pair.shape != (2,) or not torch.isfinite(pair).all():
        raise ValueError("wall_temperatures must contain two finite temperatures")
    return pair


def _surface_heater(wall_applied_flux, wall_types, like):
    heater = torch.as_tensor(wall_applied_flux, dtype=like.dtype, device=like.device)
    if heater.shape != (2,) or not torch.isfinite(heater).all():
        raise ValueError("wall_applied_flux must contain two finite inward fluxes")
    if any(kind == "dirichlet" and heater[i] != 0 for i, kind in enumerate(wall_types)):
        raise ValueError("wall_applied_flux requires a Neumann or Robin wall")
    return heater


def _wall_penalty(penalty, wall_penalty):
    """Dirichlet coefficient, independent of the interior face penalty."""
    value = penalty if wall_penalty is None else wall_penalty
    if not math.isfinite(value) or value <= 0:
        raise ValueError("wall_penalty must be finite and positive")
    return value


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


def _integrated_face_heat(coefficients, penalty, walls, duration, wall_penalty=None):
    integrated_walls = duration*_wall_pair(walls, coefficients)
    return sipg_heat_face_fluxes(coefficients, penalty, wall_temperatures=integrated_walls, wall_penalty=wall_penalty).tolist()


def _add_block(matrix, indices, block):
    matrix[indices[:, None], indices[None, :]] += block


def assemble_sipg_1d(
    ncells: int = 16,
    degree: int = 2,
    penalty: float = 20.0,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    *, wall_penalty=None, initial_temperature=None, source=None, wall_temperatures=None,
):
    """Assemble the SIPG mass/diffusion matrices for ``u_t = u_xx`` on (0,1).

    source_load includes prescribed volumetric and Dirichlet terms.
    The same operator is used by the Crank--Nicolson driver
    :func:`solve_heat_1d`, the matrix-exponential driver
    :func:`solve_heat_1d_exact_time` and any external study of the semi-discrete
    spectrum, so the discrete operator has a single source of truth.

    Returns a dict with ``centers``, ``h``, ``dofs``, ``mass_matrix`` (SPD),
    ``diffusion`` (the SPD SIPG stiffness ``K`` of ``M du/dt = -K u``, with face
    penalty ``penalty * (p+1)**2 / h`` on interior faces and the optional
    coefficient `wall_penalty` on Dirichlet faces (defaults to `penalty`)), and
    ``initial_coefficients`` (L2 projection of the supplied profile, default ``sin(pi*x)``).
    """
    wall_penalty = _wall_penalty(penalty, wall_penalty)
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
        block = consistency + consistency.T + wall_penalty * modes * modes / h * torch.outer(
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
        "source_load": _stationary_load(centers,h,operators,source,wall_penalty,wall_temperatures),
        "penalty": penalty, "wall_penalty": wall_penalty,
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
    *, wall_penalty=None, integrate_temperature: bool = False, integrate_face_fluxes: bool = False, initial_temperature=None, source=None, wall_temperatures=None,
):
    """Solve the forced heat equation with fixed Dirichlet data and a projected initial profile.

    Crank--Nicolson is used for the global method-of-lines system. If
    ``time_steps`` is omitted, dt is approximately ``0.02*cell_width``.

    Parameters
    ----------
    wall_penalty : optional positive float
        Dirichlet boundary coefficient. Defaults to the interior penalty.
    wall_temperatures : optional pair
        Constant left and right temperatures, default (0, 0).
    source : optional scalar or callable of x
        Stationary volumetric source for u_t = u_xx + source on the unit slab.
    initial_temperature : optional callable or modal coefficient tensor
        Evaluated on cell quadrature points and projected onto the modal basis.
        The default profile is sin(pi*x).
    integrate_temperature : bool
        Include time-integrated Legendre coefficients under "integrated_temperature_modes".
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
            integrate_temperature=integrate_temperature, integrate_face_fluxes=integrate_face_fluxes, initial_temperature=initial_temperature, source=source, wall_temperatures=wall_temperatures, wall_penalty=wall_penalty,
        )
    elif backend == "matrix-free":
        from tensordg.sipg_matrixfree import solve_heat_1d_matrixfree
        return solve_heat_1d_matrixfree(
            ncells, degree, final_time, time_steps, penalty, device, dtype,
            integrate_temperature=integrate_temperature, integrate_face_fluxes=integrate_face_fluxes, initial_temperature=initial_temperature, source=source, wall_temperatures=wall_temperatures, wall_penalty=wall_penalty,
        )

    # backend == "dense": original implementation
    if ncells < 2 or degree < 1 or penalty <= 0.0:
        raise ValueError("ncells >= 2, degree >= 1, and positive penalty are required")
    if final_time < 0.0 or (time_steps is not None and time_steps < 1):
        raise ValueError("final_time must be >= 0 and time_steps must be positive")
    dev = torch.device(device)
    assembly = assemble_sipg_1d(
        ncells=ncells, degree=degree, penalty=penalty, device=device, dtype=dtype, initial_temperature=initial_temperature, source=source, wall_temperatures=wall_temperatures, wall_penalty=wall_penalty,
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
    integrated = torch.zeros_like(state) if integrate_face_fluxes or integrate_temperature else None
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
        "steps": steps, "dt": dt, "penalty": penalty, "wall_penalty": _wall_penalty(penalty, wall_penalty), "dofs": ndofs,
        "diffusion_symmetry_error": float(torch.max(torch.abs(diffusion - diffusion.T)).item()),
        "device": str(dev),
    }
    if integrated is not None:
        integrated_coeff = torch.as_tensor(integrated, dtype=dtype, device=dev).reshape(ncells, modes)
        if integrate_temperature:
            metadata["integrated_temperature_modes"] = integrated_coeff.tolist()
        if integrate_face_fluxes:
            metadata["integrated_face_fluxes"] = _integrated_face_heat(integrated_coeff, penalty, wall_temperatures, final_time, wall_penalty)
    return centers, coeff, metadata


def solve_heat_1d_exact_time(
    ncells: int = 16,
    degree: int = 2,
    final_time: float = 0.1,
    penalty: float = 20.0,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    *, wall_penalty=None, initial_temperature=None, source=None, wall_temperatures=None, integrate_face_fluxes: bool = False, integrate_temperature: bool = False,
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
        ncells=ncells, degree=degree, penalty=penalty, device=device, dtype=dtype, initial_temperature=initial_temperature, source=source, wall_temperatures=wall_temperatures, wall_penalty=wall_penalty,
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
        "penalty": penalty, "wall_penalty": _wall_penalty(penalty, wall_penalty),
        "dofs": system["dofs"],
        "time_integration": "matrix-exponential",
        "mass_symmetry_error": float(np.max(np.abs(mass - mass.T))),
        "diffusion_symmetry_error": float(np.max(np.abs(diffusion - diffusion.T))),
        "stiffness_eigenvalue_min": float(eigenvalues.real.min()),
        "stiffness_eigenvalue_max": float(eigenvalues.real.max()),
        "time_step_limit": float(2.0 / eigenvalues.real.max()),
    }
    if integrate_face_fluxes or integrate_temperature:
        # Integrate M du/dt = f-Ku before recovering the numerical face flux.
        rhs = mass @ (initial-state)
        if system["source_load"] is not None:
            rhs += final_time*system["source_load"].detach().cpu().numpy().reshape(-1)
        integrated = np.linalg.solve(diffusion, rhs).reshape(ncells, degree+1)
        modes = torch.as_tensor(integrated, device=coeff.device, dtype=coeff.dtype)
        if integrate_temperature:
            metadata["integrated_temperature_modes"] = modes.tolist()
        if integrate_face_fluxes:
            metadata["integrated_face_fluxes"] = _integrated_face_heat(modes, penalty, wall_temperatures, final_time, wall_penalty)
    return system["centers"], coeff, metadata


def heat_error(
    centers: torch.Tensor, coeff: torch.Tensor, final_time: float, *, reference=exact_temperature,
) -> dict[str, float]:
    """Quadrature error against reference(x, time), defaulting to the first sine mode."""
    degree = coeff.shape[1] - 1
    h = 1.0 / centers.numel()
    nodes_np, weights_np = np.polynomial.legendre.leggauss(max(2 * degree + 5, 8))
    nodes = torch.as_tensor(nodes_np, device=coeff.device, dtype=coeff.real.dtype)
    weights = torch.as_tensor(weights_np, device=coeff.device, dtype=coeff.real.dtype)
    basis = []
    for k in range(degree + 1):
        polynomial = np.zeros(k + 1)
        polynomial[-1] = 1.0
        basis.append(torch.as_tensor(
            np.polynomial.legendre.legval(nodes_np, polynomial),
            device=coeff.device, dtype=coeff.real.dtype,
        ))
    phi = torch.stack(basis, dim=1)
    numerical = coeff @ phi.T.to(coeff.dtype)
    x = centers[:, None] + 0.5 * h * nodes[None, :]
    exact = reference(x, final_time)
    error2 = 0.5 * h * torch.sum(weights[None, :] * (numerical - exact).abs().square())
    norm2 = 0.5 * h * torch.sum(weights[None, :] * exact.abs().square())
    absolute = float(torch.sqrt(error2).item())
    norm = float(torch.sqrt(norm2).item())
    return {"absolute_l2": absolute, "exact_l2": norm, "relative_l2": absolute / norm}


def sipg_heat_face_fluxes(coefficients: torch.Tensor, penalty: float = 20., *, wall_temperatures=None, wall_penalty=None) -> torch.Tensor:
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
    boundary_strength = _wall_penalty(penalty, wall_penalty)*modes**2*cells
    return torch.cat((-grad_l[:1]-boundary_strength*(trace_l[:1]-walls[0]), interior,
                      -grad_r[-1:]+boundary_strength*(trace_r[-1:]-walls[1])))


def solve_heat_1d_ramped_walls(ncells=16, degree=2, final_time=.1, *, wall_temperatures,
                               wall_rates, initial_temperature=None, source=None, backend="dense",
                               time_steps=None, penalty=20., wall_penalty=None, device="cpu", dtype=torch.float64,
                               integrate_face_fluxes=False, integrate_temperature=False):
    """Solve with wall values g(0)+t*wall_rates, homogeneous material, and stationary source.

    The returned coefficients describe the physical temperature, and the metadata
    reports both final wall temperatures. backend="exact-time" evolves the lifted
    semi-discrete problem by matrix exponential. Other backends use Crank–Nicolson.
    """
    like = torch.empty(0, device=device, dtype=dtype)
    walls = _wall_pair(wall_temperatures, like)
    rates = _wall_pair(wall_rates, like)
    slope = rates[1]-rates[0]

    def lifted_source(x):
        generation = source(x) if callable(source) else (0. if source is None else source)
        return generation-(rates[0]+slope*x)

    options = dict(ncells=ncells,degree=degree,final_time=final_time,wall_temperatures=walls,
                   initial_temperature=initial_temperature,source=lifted_source,penalty=penalty,wall_penalty=wall_penalty,
                   device=device,dtype=dtype,integrate_face_fluxes=integrate_face_fluxes, integrate_temperature=integrate_temperature)
    if backend == "exact-time":
        if time_steps is not None:
            raise ValueError("time_steps applies only to Crank–Nicolson backends")
        centers, coeff, meta = solve_heat_1d_exact_time(**options)
    else:
        centers, coeff, meta = solve_heat_1d(**options,backend=backend,time_steps=time_steps)
    # The spatially affine lift occupies only the constant and linear Legendre modes.
    lift = torch.zeros_like(coeff)
    lift[:,0] = final_time*(rates[0]+slope*centers)
    lift[:,1] = final_time*slope/(2*ncells)
    if integrate_temperature:
        integrated = coeff.new_tensor(meta["integrated_temperature_modes"])
        meta["integrated_temperature_modes"] = (integrated + 0.5*final_time*lift).tolist()
    meta["final_wall_temperatures"] = (walls+final_time*rates).tolist()
    if integrate_face_fluxes:
        heat = coeff.new_tensor(meta["integrated_face_fluxes"])
        meta["integrated_face_fluxes"] = (heat-.5*final_time**2*slope).tolist()
    return centers, coeff+lift, meta


def solve_heat_1d_wall_history(times, wall_temperatures, ncells=16, degree=2, *,
                               initial_temperature=None, source=None, interval_sources=None, backend="dense",
                               steps_per_interval=None, penalty=20., wall_penalty=None, device="cpu",
                               dtype=torch.float64, integrate_face_fluxes=False, return_interval_temperatures=False):
    """Evolve piecewise-linear left/right wall temperatures on a uniform unit slab.

    Supply wall_temperatures as (knots, 2) for continuous histories or
    (intervals, 2, 2) for jumps. Each interval then specifies its start and end
    (left, right) pairs independently; the modal state is continuous at a jump.
    times are strictly increasing knot times. The initial field is defined at
    the first knot. Return centers, modal fields at every knot (including the
    initial field), and metadata. interval_face_heat contains the rightward
    transfer at each face over each individual interval, when requested.
    return_interval_temperatures adds interval_temperature_modes to metadata,
    the time-mean Legendre coefficients with shape (interval, cell, mode).
    steps_per_interval is the number of CN steps per segment, or None to use
    the solver default. Use backend="exact-time" for semi-discrete exponential
    evolution. Supply a stationary source, or interval_sources with one
    scalar or callable of x per interval. None denotes zero generation.
    """
    times = torch.as_tensor(times, dtype=torch.float64)
    walls = torch.as_tensor(wall_temperatures, dtype=dtype, device=device)
    if times.ndim != 1 or times.numel() < 2 or not torch.isfinite(times).all() or (times.diff() <= 0).any():
        raise ValueError("times must contain at least two finite, strictly increasing knot times")
    if walls.shape == (times.numel(), 2):
        start, end = walls[:-1], walls[1:]
    elif walls.shape == (times.numel()-1, 2, 2):
        start, end = walls[:, 0], walls[:, 1]
    else:
        raise ValueError("wall_temperatures requires (knots, 2) or (intervals, 2, 2)")
    if not torch.isfinite(walls).all():
        raise ValueError("wall_temperatures must be finite")
    if ncells < 2 or degree < 1 or not math.isfinite(penalty) or penalty <= 0:
        raise ValueError("ncells >= 2, degree >= 1 and finite positive penalty are required")
    if steps_per_interval is not None and (not isinstance(steps_per_interval, int) or steps_per_interval < 1):
        raise ValueError("steps_per_interval must be a positive integer")
    if interval_sources is not None:
        if source is not None:
            raise ValueError("provide source or interval_sources, not both")
        if len(interval_sources) != times.numel()-1:
            raise ValueError("interval_sources must contain one profile per interval")
    centers = (torch.arange(ncells, dtype=dtype, device=device)+.5)/ncells
    nodes, weights, phi, *_ = _operators(degree, centers.device, dtype)
    state = _project_initial(centers, 1/ncells, nodes, weights, phi, initial_temperature)
    history = [state]
    transfers = []
    means = []
    intervals = []
    for i, duration in enumerate(times.diff().tolist()):
        rates = (end[i]-start[i])/duration
        generation = source if interval_sources is None else interval_sources[i]
        _, state, meta = solve_heat_1d_ramped_walls(
            ncells, degree, duration, wall_temperatures=start[i], wall_rates=rates,
            initial_temperature=state, source=generation, backend=backend, time_steps=steps_per_interval,
            penalty=penalty, wall_penalty=wall_penalty, device=device, dtype=dtype, integrate_face_fluxes=integrate_face_fluxes,
            integrate_temperature=return_interval_temperatures)
        history.append(state)
        if integrate_face_fluxes:
            transfers.append(meta.pop("integrated_face_fluxes"))
        if return_interval_temperatures:
            means.append((state.new_tensor(meta.pop("integrated_temperature_modes"))/duration).tolist())
        intervals.append(meta)
    metadata = {"times": times.tolist(), "intervals": intervals}
    if integrate_face_fluxes:
        heat = state.new_tensor(transfers)
        metadata["interval_face_heat"] = transfers
        metadata["integrated_face_fluxes"] = heat.sum(dim=0).tolist()
    if return_interval_temperatures:
        metadata["interval_temperature_modes"] = means
    return centers, torch.stack(history), metadata


def _bulk_profile(profile, points, dtype):
    raw = profile(points) if callable(profile) else profile
    if dtype == torch.float64 and torch.is_complex(torch.as_tensor(raw)):
        raise ValueError("real bulk profiles are required")
    values = torch.as_tensor(raw,dtype=dtype)
    if not callable(profile) and values.shape == (len(points),):
        values = values[:,None]
    if not torch.isfinite(values).all():
        raise ValueError("bulk exchange profiles must be finite")
    return values.expand_as(points)


def _bulk_exchange_matrix(loss, weights, phi, h):
    from scipy import sparse

    if torch.any(loss<0):
        raise ValueError("heat_loss_coefficient must be nonnegative")
    weighted = (.5*h)*loss*weights
    if not torch.any(loss):
        return None,weighted
    blocks = torch.einsum("eq,qi,qj->eij",weighted,phi,phi)
    n = len(loss)
    return sparse.bsr_matrix((blocks.detach().numpy(),np.arange(n),np.arange(n+1))).tocsr(),weighted


def _slab_trace_matrix(ncells, degree, left, right):
    from scipy import sparse

    modes = degree+1
    columns = np.r_[np.arange(modes),np.arange((ncells-1)*modes,ncells*modes)]
    return sparse.csr_matrix((np.r_[left.numpy(),right.numpy()],columns,[0,modes,2*modes]),
                             shape=(2,ncells*modes))


def solve_harmonic_heat_1d(angular_frequency, *, ncells=16, degree=2, conductivity=1., capacity=1.,
                           source=None, wall_types=("dirichlet","dirichlet"), wall_values=(0.,0.), penalty=20.,
                           film_coefficients=(0.,0.), wall_areal_heat_capacity=(0.,0.),
                           return_wall_temperatures=False, heat_loss_coefficient=None, bulk_ambient_temperature=0.,
                           separate_mean=False, return_mean_rate=False, return_cell_power=False, return_wall_power=False,
                           wall_applied_flux=(0., 0.)):
    """Solve iωcU - kU'' = source on the unit slab with exp(+iωt) convention.

    Each end prescribes a temperature (dirichlet), inward external heat flux
    (neumann), or ambient temperature (robin). film_coefficients and
    wall_areal_heat_capacity are left/right pairs in W/(m² K) and J/(m² K).
    wall_applied_flux adds an independent inward surface heater (W/m²,
    left/right phasors) to a Neumann or Robin wall.
    A flux or Robin wall can store heat. The material receives external input
    minus iω*Cwall*Twall. Set return_wall_temperatures to append both wall
    temperatures (prescribed on Dirichlet walls, solved traces elsewhere).
    The optional bulk exchange H*(Tambient-U) enters as a volumetric source.
    H in W/(m³ K) is finite, real and nonnegative. H and Tambient accept
    constants, one value per cell, or functions of physical quadrature points.
    Source phasors may be constant or callable on real cell quadrature points.
    Returns cell centers, complex128 Legendre coefficients and the rightward
    numerical flux at every face. Uses a CPU sparse direct solve.
    return_cell_power appends a dict after optional wall temperatures and
    before any mean or rate. Its source, bulk and storage entries are complex
    cell-integrated powers per unit area (W/m²). Positive bulk exchange heats
    the material; storage equals source + bulk + left flux - right flux.
    return_wall_power appends a dict after any cell powers and before the mean
    or rate. Each entry is a left/right pair in W/m²: external power into the
    wall, storage in its areal capacity, and material power into the slab.
    The applied and film entries isolate the extra heater and Robin exchange.
    At a Dirichlet wall external power is the numerical boundary reaction.
    With separate_mean=True on a slab without temperature, film or bulk
    exchange anchors, return temperature coefficients and optional wall traces
    relative to the capacity-weighted mean (including wall storage). Append
    that uniform phasor in K as the last result. Fluxes retain physical values.
    Alternatively, return_mean_rate=True appends the mean heating rate in K/s
    and returns the same relative temperatures. This mode also supports zero
    frequency: under constant loads the absolute temperature is the stationary
    relative profile plus a spatially uniform temperature rising at this rate.
    """
    from scipy import sparse
    from scipy.sparse.linalg import splu
    from .sipg_sparse import assemble_heat_1d_sparse

    if ncells < 2 or degree < 1:
        raise ValueError("ncells >= 2 and degree >= 1 are required")
    if not math.isfinite(angular_frequency) or angular_frequency < 0:
        raise ValueError("frequency must be finite and nonnegative")
    if angular_frequency == 0 and not return_mean_rate:
        raise ValueError("zero frequency requires return_mean_rate")
    if separate_mean and return_mean_rate:
        raise ValueError("choose either separate_mean or return_mean_rate")
    if not all(math.isfinite(v) and v > 0 for v in (conductivity,capacity,penalty)):
        raise ValueError("conductivity, capacity and penalty must be positive and finite")
    if len(wall_types)!=2 or any(kind not in ("dirichlet","neumann","robin") for kind in wall_types):
        raise ValueError("each wall type must be dirichlet, neumann or robin")
    film = np.asarray(film_coefficients)
    wall_capacity = np.asarray(wall_areal_heat_capacity)
    for data in (film,wall_capacity):
        if data.shape!=(2,) or np.iscomplexobj(data) or not np.isfinite(data).all() or (data<0).any():
            raise ValueError("film and wall capacity require finite nonnegative real pairs")
    kinds = np.asarray(wall_types)
    if np.any(film[kinds!="robin"]) or np.any(wall_capacity[kinds=="dirichlet"]):
        raise ValueError("films require Robin walls and storage requires flux or Robin walls")
    h = 1./ncells
    centers = (torch.arange(ncells,dtype=torch.float64)+.5)*h
    stiffness,mass,ops = assemble_heat_1d_sparse(ncells,degree,penalty,centers.device,centers.dtype,
                                                 wall_types=tuple("neumann" if kind=="robin" else kind for kind in wall_types))
    nodes,weights,phi,_,_,left,right,dleft,dright = ops
    load = torch.zeros((ncells,degree+1),dtype=torch.complex128) if source is None else _source_load(
        centers,h,nodes,weights,phi,source,dtype=torch.complex128)
    if return_cell_power:
        source_power = load[:, 0].clone()
    exchange = None
    if heat_loss_coefficient is not None:
        points = centers[:,None] + .5*h*nodes
        loss = _bulk_profile(heat_loss_coefficient,points,torch.float64)
        exchange,weighted = _bulk_exchange_matrix(loss,weights,phi,h)
        ambient = _bulk_profile(bulk_ambient_temperature,points,torch.complex128)
        load += (weighted*ambient) @ phi.to(load.dtype)
    values = _wall_pair(wall_values,load)
    heater = _surface_heater(wall_applied_flux, wall_types, load)
    strength = penalty*(degree+1)**2/h
    for i,(cell,v,dv,normal) in enumerate(((0,left,dleft,-1),(-1,right,dright,1))):
        boundary = conductivity*(strength*v-normal*2*dv/h) if wall_types[i]=="dirichlet" else v
        applied = film[i]*values[i] if wall_types[i]=="robin" else values[i]
        load[cell] += (applied+heater[i])*boundary
    storage = 1j*angular_frequency*capacity
    system = conductivity*stiffness + storage*mass
    if exchange is not None:
        system += exchange
    trace = _slab_trace_matrix(ncells,degree,left,right)
    if np.any(film) or np.any(wall_capacity):
        system += trace.T @ sparse.diags(film+1j*angular_frequency*wall_capacity) @ trace
    rhs = load.detach().numpy().ravel()
    mean = rate = 0j
    unanchored = exchange is None and "dirichlet" not in wall_types and not np.any(film)
    if (separate_mean or return_mean_rate) and not unanchored:
        raise ValueError("mean separation requires a slab without temperature or exchange anchors")
    if unanchored:
        # Constant mode has no conductive restoring force. Separate its exact
        # storage response and constrain the remaining fluctuation to zero mean.
        integral = np.zeros_like(rhs)
        integral[::degree+1] = capacity*h
        integral += trace.T @ wall_capacity
        rate = rhs[::degree+1].sum()/(capacity+wall_capacity.sum())
        if not return_mean_rate:
            mean = rate/(1j*angular_frequency)
        column = sparse.csc_matrix(integral[:,None])
        bordered = sparse.bmat([[system,column],[column.T,None]],format="csc")
        state = splu(bordered).solve(np.r_[rhs-rate*integral,0j])[:-1]
    else:
        factor = splu(system.tocsc())
        state = factor.solve(rhs)
        # Evaluate the weak-form residual before combining diffusion, volume
        # storage and surface terms. Small wall admittances otherwise lose
        # precision when added to the much larger penalty entries.
        trial = state.astype(np.clongdouble)
        residual = rhs - conductivity*(stiffness @ trial) - storage*(mass @ trial)
        if exchange is not None:
            residual -= exchange @ trial
        admittance = film+1j*angular_frequency*wall_capacity
        residual -= trace.T @ (admittance*(trace @ trial))
        state += factor.solve(residual.astype(np.complex128))
    coefficients = torch.from_numpy(state.reshape(ncells,degree+1))
    flux = conductivity*sipg_heat_face_fluxes(coefficients,penalty,wall_temperatures=values)
    surface = torch.from_numpy(trace @ state)
    if return_wall_power:
        wall_power = {name: torch.empty_like(surface) for name in ("external", "storage", "material")}
        wall_power.update(applied=heater.clone(), film=torch.zeros_like(surface))
    for i,face in enumerate((0,-1)):
        if wall_types[i]=="dirichlet":
            surface[i] = values[i]
            external = (1-2*i)*flux[face]
            stored = 0.
        else:
            if wall_types[i]=="robin":
                film_power = film[i]*(values[i]-surface[i])
                external = film_power + heater[i]
                if return_wall_power:
                    wall_power["film"][i] = film_power
            else:
                external = values[i] + heater[i]
            common_storage = wall_capacity[i]*rate
            relative_storage = (1j*angular_frequency*wall_capacity[i])*surface[i]
            stored = common_storage + relative_storage
            # Subtract the common heating first to retain the smaller relative
            # storage term when the prescribed input balances the common part.
            flux[face] = (1-2*i)*(external-common_storage-relative_storage)
        if return_wall_power:
            wall_power["external"][i] = external
            wall_power["storage"][i] = stored
            wall_power["material"][i] = (1-2*i)*flux[face]
    if return_cell_power:
        # For floating slabs the solve returns a relative state. Compute its
        # storage before adding the uniform phasor, which can be huge at low ω.
        storage_power = capacity*h*(rate + 1j*angular_frequency*coefficients[:, 0])
        bulk_power = torch.zeros_like(source_power)
        if exchange is not None:
            field = coefficients @ phi.to(coefficients.dtype).T
            bulk_power = (weighted*(ambient-field)).sum(dim=1)
        cell_power = dict(source=source_power, bulk=bulk_power, storage=storage_power)
    if not (separate_mean or return_mean_rate):
        coefficients[:,0] += mean
        surface += mean
    result = (centers,coefficients,flux)
    if return_wall_temperatures:
        result += (surface,)
    if return_cell_power:
        result += (cell_power,)
    if return_wall_power:
        result += (wall_power,)
    if return_mean_rate:
        return (*result, coefficients.new_tensor(rate))
    return (*result, coefficients.new_tensor(mean)) if separate_mean else result


def sample_heat_stack_fields(coefficients, lengths, conductivity, fractions):
    """Sample DG temperature and rightward Fourier flux within each layer cell.

    coefficients has shape (layers, cells_per_layer, modes), as returned by
    solve_drifting_heat_stack_1d. fractions is a vector of cell coordinates
    from 0 (west) to 1 (east). Both outputs have shape (layers, cells, samples).
    Flux is the derivative-based material estimate in W/m². At endpoints
    the solve also supplies the conservative SIPG numerical face flux.
    """
    from .curved_geometry import legendre_derivative_matrix, legendre_eval_matrix

    u = torch.as_tensor(coefficients)
    if u.ndim != 3 or min(u.shape) < 1 or not (u.is_floating_point() or u.is_complex()):
        raise ValueError("provide floating-point modal coefficients shaped (layers, cells, modes)")
    f = np.asarray(fractions)
    if f.ndim != 1 or not f.size or np.iscomplexobj(f) or not np.all((f >= 0) & (f <= 1)):
        raise ValueError("provide a nonempty vector of cell fractions in [0, 1]")
    layers, cells, modes = u.shape
    h = torch.as_tensor(lengths, device=u.device)
    k = torch.as_tensor(conductivity, device=u.device)
    for values in (h, k):
        if values.is_complex() or not torch.isfinite(values).all() or (values <= 0).any():
            raise ValueError("layer lengths and conductivity must be finite, real and positive")
    if h.shape != (layers,) or k.shape not in ((), (layers,)):
        raise ValueError("provide a length per layer and scalar or per-layer conductivity")
    # Convert the original inputs at field precision, including Python lists.
    h = torch.as_tensor(lengths, dtype=u.real.dtype, device=u.device)
    k = torch.as_tensor(conductivity, dtype=u.real.dtype, device=u.device)
    basis = legendre_eval_matrix(modes-1, 2.*f-1.)
    derivative = basis @ legendre_derivative_matrix(modes-1)
    temperature = u @ u.new_tensor(basis.T)
    flux = (-2*cells*k/h)[:, None, None]*(u @ u.new_tensor(derivative.T))
    return temperature, flux


def _stack_flux_polynomial(face_flux, lengths, fractions, source_gradient, source_curvature):
    q = torch.as_tensor(face_flux)
    if q.ndim != 2 or q.shape[0] < 1 or q.shape[1] < 2 or not q.is_floating_point():
        raise ValueError("provide real floating-point fluxes shaped (layers, cells+1)")
    if not torch.isfinite(q).all():
        raise ValueError("face fluxes must be finite")

    def real_data(value):
        if torch.is_complex(torch.as_tensor(value)):
            raise ValueError("lengths, fractions and source gradients must be real")
        return torch.as_tensor(value, dtype=q.dtype, device=q.device)

    length, f, g, c = map(real_data, (lengths, fractions, source_gradient, source_curvature))
    if length.shape != (q.shape[0],) or not torch.isfinite(length).all() or (length <= 0).any():
        raise ValueError("provide one finite positive length per layer")
    if g.shape not in ((), length.shape) or not torch.isfinite(g).all():
        raise ValueError("provide a finite scalar or per-layer source gradient")
    if f.ndim != 1 or not f.numel() or not ((f >= 0) & (f <= 1)).all():
        raise ValueError("provide a nonempty vector of cell fractions in [0, 1]")
    width = length/(q.shape[1]-1)
    if c.shape not in ((), length.shape) or not torch.isfinite(c).all():
        raise ValueError("provide a finite scalar or per-layer source curvature")
    cells = q.shape[1]-1
    centers = (torch.arange(cells, dtype=q.dtype, device=q.device)+.5-cells/2)*width[:, None]
    local_gradient = g.reshape(-1, 1) + c.reshape(-1, 1)*centers
    quadratic = (.5*width[:, None].square()*local_gradient)[:, :, None]
    cubic = (c*width**3/6)[:, None, None]
    return q, f, width, quadratic, cubic


def reconstruct_heat_stack_flux(face_flux, lengths, fractions, *, source_gradient=0., source_curvature=0.):
    """Recover conservative rightward flux density inside drifting stack cells.

    Supply face_flux (W/m²) from solve_drifting_heat_stack_1d and the same
    layer lengths (m) and source_gradient (W/m⁴) and source_curvature (W/m⁵). The uniform heating rate
    makes dq/dx = source - capacity*rate at most quadratic in each homogeneous layer.
    Each cell's two numerical fluxes fix the integration constants and its
    source derivatives fix the interior polynomial. Each layer owns its contact traces.
    The result is shaped (layers, cells, samples) at fractions from 0 to 1.
    """
    q, f, _, curvature, cubic = _stack_flux_polynomial(face_flux, lengths, fractions, source_gradient, source_curvature)
    return q[:, :-1, None]*(1-f) + q[:, 1:, None]*f + f*(f-1)*(curvature + cubic*(f-.5))


def reconstruct_heat_stack_temperature(cell_means, face_flux, lengths, conductivity, fractions, *,
                                       source_gradient=0., source_curvature=0.):
    """Integrate the conservative stack flux and retain each input cell mean.

    Supply the zeroth DG modes in K (layers, cells), conductivities in W/m/K,
    and the same data as reconstruct_heat_stack_flux. The resulting at-most-quartic
    temperature uses the gauge of cell_means and keeps separate contact traces.
    """
    q, f, width, curvature, cubic = _stack_flux_polynomial(face_flux, lengths, fractions, source_gradient, source_curvature)
    if torch.is_complex(torch.as_tensor(cell_means)) or torch.is_complex(torch.as_tensor(conductivity)):
        raise ValueError("cell means and conductivity must be real")
    mean = torch.as_tensor(cell_means, dtype=q.dtype, device=q.device)
    k = torch.as_tensor(conductivity, dtype=q.dtype, device=q.device)
    if mean.shape != q[:, :-1].shape or not torch.isfinite(mean).all():
        raise ValueError("provide a finite mean temperature for every cell")
    if k.shape not in ((), width.shape) or not torch.isfinite(k).all() or (k <= 0).any():
        raise ValueError("provide a positive finite scalar or per-layer conductivity")
    # Each basis term integrates to zero over f in [0,1]. Its derivative
    # recovers one term of the quadratic flux in local cell coordinates.
    west = q[:, :-1, None]
    slope = q.diff(dim=1)[:, :, None]-curvature
    drop = west*(f-.5) + slope*(f.square()-1/3)/2 + curvature*(f**3-.25)/3
    drop += cubic*(f.square()*(1-f).square()/4 - 1/120)
    return mean[:, :, None]-(width/k)[:, None, None]*drop


def solve_drifting_heat_stack_1d(lengths, conductivity, capacity, wall_inward_flux, *,
                                 source=0., source_gradient=0., source_curvature=0., contact_resistance=0., area=1., ncells=16, degree=2,
                                 shoulder_areal_heat_capacity=0., end_areal_heat_capacity=0.,
                                 interface_left_areal_heat_capacity=0., interface_areal_heat_capacity=0.,
                                 interface_left_heat_sources=0., interface_heat_sources=0.):
    """Stationary relative temperature of a stepped stack under constant loads.

    lengths (m), conductivity (W/m/K) and volumetric capacity (J/m³/K)
    describe homogeneous layers. source (W/m³) is the layer mean. Within
    each layer the generation is source + source_gradient*(x-x_mid), with
    gradient in W/m⁴ and x_mid the physical layer midpoint. Optional
    source_curvature (W/m⁵) adds 0.5*source_curvature*((x-x_mid)²-length²/12),
    preserving the prescribed layer mean.
    area (m²) is constant per layer. Contacts cover the smaller adjoining
    section. The insulated shoulder stores shoulder_areal_heat_capacity
    (J/m²/K) times the exposed area at the larger section’s trace temperature.
    Contact data use the overlap area, external wall fluxes use the local
    end areas, and returned flux densities use their layer areas.
    Each contact obeys T_left-T_right=R*q for the crossing flux, R in m² K/W.
    interface_left_areal_heat_capacity and interface_areal_heat_capacity
    store heat on its left and right traces respectively (J/m²/K). The
    returned face fluxes are inside the material on each side of the contact.
    interface_left_heat_sources and interface_heat_sources inject heat on the
    left and right traces respectively (W/m², negative for extraction).
    end_areal_heat_capacity (J/m²/K, scalar or left/right pair) stores heat
    at the external wall traces. Applied wall flux is positive inward and
    includes both end storage and the material flux. Scalars for material,
    source or contact data apply throughout. ncells is the count per layer.
    Return physical centers, Legendre coefficients, rightward face fluxes
    and both end traces per layer (layer index first), then the common heating
    rate in K/s. The relative field has zero integrated stored energy.
    """
    lengths = np.asarray(lengths)
    if lengths.ndim != 1 or lengths.size == 0:
        raise ValueError("provide one or more layer lengths")
    def profile(value, size, name, *, positive=False):
        arr = np.asarray(value)
        if np.iscomplexobj(arr) or arr.shape not in ((), (size,)) or not np.isfinite(arr).all():
            raise ValueError(f"provide finite real {name} with the requested shape")
        if positive and np.any(arr <= 0):
            raise ValueError(f"{name} must be positive")
        return np.broadcast_to(arr.astype(float), (size,))

    layers = lengths.size
    lengths = profile(lengths, layers, "lengths", positive=True)
    k = profile(conductivity, layers, "conductivity", positive=True)
    c = profile(capacity, layers, "capacity", positive=True)
    generation = profile(source, layers, "source")
    gradients = profile(source_gradient, layers, "source gradient")
    curvatures = profile(source_curvature, layers, "source curvature")
    resistance = profile(contact_resistance, layers-1, "contact resistance")
    if np.any(resistance < 0):
        raise ValueError("contact resistance must be nonnegative")
    if np.shape(wall_inward_flux) != (2,):
        raise ValueError("provide two inward wall fluxes")
    west, east = profile(wall_inward_flux, 2, "wall fluxes")
    area = profile(area, layers, "section area", positive=True)
    overlap = np.minimum(area[:-1], area[1:])
    west, east = west*area[0], east*area[-1]
    volume = lengths*area
    mass = volume*c
    left = profile(interface_left_areal_heat_capacity, layers-1, "left contact capacity")
    right = profile(interface_areal_heat_capacity, layers-1, "right contact capacity")
    if np.any(left < 0) or np.any(right < 0):
        raise ValueError("contact capacities must be nonnegative")
    wall_mass = np.column_stack((np.r_[0., right*overlap], np.r_[left*overlap, 0.]))
    end_capacity = profile(end_areal_heat_capacity, 2, "end capacity")
    if np.any(end_capacity < 0):
        raise ValueError("end capacity must be nonnegative")
    wall_mass[0, 0], wall_mass[-1, 1] = end_capacity*area[[0, -1]]
    shoulder = profile(shoulder_areal_heat_capacity, layers-1, "shoulder capacity")
    if np.any(shoulder < 0):
        raise ValueError("shoulder capacity must be nonnegative")
    area_step = np.diff(area)
    wall_mass[1:, 0] += shoulder*np.maximum(area_step, 0.)
    wall_mass[:-1, 1] += shoulder*np.maximum(-area_step, 0.)
    layer_mass = mass+wall_mass.sum(axis=1)
    left_load = profile(interface_left_heat_sources, layers-1, "left contact source")
    right_load = profile(interface_heat_sources, layers-1, "right contact source")
    wall_load = np.column_stack((np.r_[0., right_load*overlap], np.r_[left_load*overlap, 0.]))
    layer_input = volume*generation + wall_load.sum(axis=1)
    rate = (west+east+layer_input.sum())/layer_mass.sum()
    # Layer energy balance fixes every crossing power before choosing a gauge.
    cuts = np.r_[west, west+np.cumsum(layer_input-layer_mass*rate)]
    cuts[-1] = -east
    centers, modes, fluxes, traces = [], [], [], []
    origin = 0.
    for i, length in enumerate(lengths):
        def local_load(x):
            return volume[i]*(generation[i]+gradients[i]*length*(x-.5)
                              + .5*curvatures[i]*length**2*((x-.5)**2-1/12))

        x, u, q, walls, _ = solve_harmonic_heat_1d(
            0., ncells=ncells, degree=degree, conductivity=k[i]*area[i]/length, capacity=mass[i],
            source=local_load, wall_types=("neumann", "neumann"),
            wall_values=(cuts[i]+wall_load[i, 0], -cuts[i+1]+wall_load[i, 1]),
            wall_areal_heat_capacity=wall_mass[i],
            return_mean_rate=True, return_wall_temperatures=True)
        if i:
            offset = traces[-1][1]-resistance[i-1]*cuts[i]/overlap[i-1]-walls[0]
            u[:, 0] += offset
            walls += offset
        centers.append(origin+length*x)
        modes.append(u.real)
        fluxes.append(q.real/area[i])
        traces.append(walls.real)
        origin += length
    u = torch.stack(modes)
    traces = torch.stack(traces)
    energy = torch.dot(u[:, :, 0].mean(dim=1), u.new_tensor(mass))
    shift = (energy+(traces*u.new_tensor(wall_mass)).sum())/layer_mass.sum()
    u[:, :, 0] -= shift
    return torch.stack(centers), u, torch.stack(fluxes), traces-shift, float(rate)


def harmonic_heat_wall_admittance_1d(angular_frequency, *, ncells=16, degree=2,
                                     conductivity=1., capacity=1., penalty=20.,
                                     heat_loss_coefficient=None, thickness=1., source=None):
    """Map two prescribed wall-temperature phasors to inward heat fluxes.

    For a SIPG slab of the specified thickness in meters, return a complex128 matrix Y with rows
    and columns ordered left, right. Y @ temperatures gives the pair of
    inward fluxes in W/m² for an unforced layer. If source is supplied
    (W/m³, scalar, per-cell values or a function of physical position),
    return (Y, offset): inward flux = Y @ temperatures + offset. The bulk
    reservoir temperature is zero. The frequency and material arguments
    follow solve_harmonic_heat_1d, including spatially varying bulk loss. Loss callbacks receive
    physical coordinates measured from the left wall.
    """
    if not math.isfinite(thickness) or thickness <= 0:
        raise ValueError("thickness must be positive and finite")
    # Map x=L*xi: conduction scales as k/L, volumetric terms as L.
    # Inward wall fluxes retain their physical units under this change.
    def scaled_loss(xi):
        return thickness*_bulk_profile(heat_loss_coefficient, thickness*xi, torch.float64)

    loss = None if heat_loss_coefficient is None else scaled_loss

    def inward_flux(drive, generation=None):
        _, _, flux = solve_harmonic_heat_1d(
            angular_frequency, ncells=ncells, degree=degree, conductivity=conductivity/thickness,
            capacity=capacity*thickness, penalty=penalty, heat_loss_coefficient=loss,
            wall_values=drive, source=generation)
        return torch.stack((flux[0], -flux[-1]))

    matrix = torch.stack((inward_flux((1., 0.)), inward_flux((0., 1.))), dim=1)
    if source is None:
        return matrix

    def scaled_source(xi):
        return thickness*_bulk_profile(source, thickness*xi, torch.complex128)

    return matrix, inward_flux((0., 0.), scaled_source)


def compose_heat_wall_admittances_1d(admittances, contact_resistance=0., *, wall_temperatures=None,
                                     interface_heat_sources=None, interface_left_heat_sources=None, interface_areal_heat_capacity=None,
                                     interface_left_areal_heat_capacity=None, angular_frequency=None, interface_film_coefficients=0.,
                                     interface_ambient_temperatures=0., interface_left_film_coefficients=0.,
                                     interface_left_ambient_temperatures=0., layer_flux_offsets=None,
                                     return_flux_offset=False):
    """Return the complex wall admittance of a stack, ordered left to right.

    Supply one 2-by-2 inward-flux matrix per layer at a common
    frequency and cross section. Each interface without exchange, generation or storage conserves heat flux and
    supports the temperature drop R*q (q positive to the right). R is an
    areal resistance in m² K/W, scalar or one value per interface.

    With wall_temperatures=(left, right), return (Y, temperatures, fluxes).
    The latter two complex arrays are shaped (layers, 2) and give each
    layer's left and right surface temperatures and inward heat fluxes.
    Temperatures are in K and fluxes in W/m². With prescribed wall temperatures,
    interface_heat_sources supplies heat generation (W/m², scalar or one
    complex phasor per contact) immediately to the right of each resistance.
    interface_left_heat_sources independently loads the left trace in W/m².
    With left input S, the flux crossing the resistance is S plus the
    rightward flux exiting the left material plus left film input minus the left trace storage rate.
    Optional interface_areal_heat_capacity (J/(m² K), scalar or one per
    contact) places storage at the right-hand trace. interface_left_areal_heat_capacity
    places independent storage at the left-hand trace in the same units. Supply the common
    angular_frequency (rad/s) of the layer matrices for storing interfaces.
    At each right-hand trace a film coefficient h (W/(m² K)) and ambient
    temperature phasor Ta (K) add the input h*(Ta-T_right). Supply either
    scalars or one value per interface. interface_left_film_coefficients and
    interface_left_ambient_temperatures define an independent film on the left.
    The rightward material flux increases by both trace sources and film inputs
    minus i*omega*(C_left*T_left + C_right*T_right).
    Y maps temperature perturbations at fixed sources; fluxes include the
    full driven response. To include volumetric forcing, supply each layer
    with its pair of zero-wall-temperature inward fluxes in layer_flux_offsets
    (shape layers by 2, W/m²). Set return_flux_offset=True to append the
    two exterior inward fluxes b at zero wall temperatures. Without a drive
    this returns (Y, b), so any new temperature pair T gives Y @ T + b
    while all layer and interface loads remain fixed.
    """
    layers = tuple(admittances)
    if not layers:
        raise ValueError("supply at least one layer admittance")
    matrices = [torch.as_tensor(layer, dtype=torch.complex128) for layer in layers]
    if any(layer.shape != (2, 2) or not torch.isfinite(layer).all() for layer in matrices):
        raise ValueError("layer admittances must be finite 2-by-2 matrices")
    count = len(matrices)-1
    device = matrices[0].device

    def real_parameter(value, name):
        if torch.as_tensor(value).is_complex():
            raise ValueError(f"{name} must be real")
        values = torch.as_tensor(value, dtype=torch.float64, device=device)
        if values.shape not in ((), (count,)) or not torch.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"supply a nonnegative finite {name} or one per interface")
        return values

    def film_parameters(coefficients, temperatures):
        film = real_parameter(coefficients, "film coefficient")
        ambient = torch.as_tensor(temperatures, dtype=matrices[0].dtype, device=device)
        if ambient.shape not in ((), (count,)) or not torch.isfinite(ambient).all():
            raise ValueError("supply a finite ambient temperature or one per interface")
        if count == 0 and film.any():
            raise ValueError("interface films require at least two layers")
        return film, film*ambient

    resistance = real_parameter(contact_resistance, "contact resistance")
    film, film_input = film_parameters(interface_film_coefficients, interface_ambient_temperatures)
    left_film, left_input = film_parameters(interface_left_film_coefficients, interface_left_ambient_temperatures)
    if wall_temperatures is not None:
        drive = torch.as_tensor(wall_temperatures, device=matrices[0].device, dtype=torch.complex128)
        if drive.shape != (2,) or not torch.isfinite(drive).all():
            raise ValueError("supply two finite wall temperature phasors")

    def source_profile(values):
        if values is None:
            return matrices[0].new_zeros(count)
        if wall_temperatures is None and not return_flux_offset:
            raise ValueError("interface heat sources require wall temperatures or return_flux_offset")
        sources = torch.as_tensor(values, dtype=matrices[0].dtype, device=device)
        if sources.shape not in ((), (len(layers)-1,)) or not torch.isfinite(sources).all():
            raise ValueError("supply finite heat generation or one value per interface")
        if len(layers) == 1 and sources.any():
            raise ValueError("interface heating requires at least two layers")
        return sources.expand(count)

    sources = source_profile(interface_heat_sources)
    left_sources = source_profile(interface_left_heat_sources)
    if angular_frequency is not None and (not math.isfinite(angular_frequency) or angular_frequency <= 0):
        raise ValueError("angular_frequency must be finite and positive")

    def storage_admittance(value):
        capacity = real_parameter(0. if value is None else value, "areal heat capacity")
        if capacity.any():
            if len(layers) == 1 or angular_frequency is None:
                raise ValueError("interface storage requires two or more layers and an angular frequency")
            return 1j*angular_frequency*capacity.expand(count)
        return matrices[0].new_zeros(count)

    left_admittance = storage_admittance(interface_left_areal_heat_capacity) + left_film
    storage = storage_admittance(interface_areal_heat_capacity)
    sources = sources + film_input
    left_sources = left_sources + left_input
    interface_admittance = storage + film
    offsets = matrices[0].new_zeros((len(layers), 2))
    if layer_flux_offsets is not None:
        if wall_temperatures is None and not return_flux_offset:
            raise ValueError("layer flux offsets require wall temperatures or return_flux_offset")
        pairs = [torch.as_tensor(pair, dtype=offsets.dtype, device=device) for pair in layer_flux_offsets]
        if len(pairs) != len(layers) or any(pair.shape != (2,) for pair in pairs):
            raise ValueError("supply one pair of inward flux offsets per layer")
        offsets = torch.stack(pairs)
        if not torch.isfinite(offsets).all():
            raise ValueError("layer flux offsets must be finite")
    left_flux_offset, right_flux_offset = offsets[0]
    interfaces = []
    combined = matrices[0]
    for layer, r, source, left_source, left_shunt, shunt, layer_offset in zip(
            matrices[1:], resistance.expand(count), sources, left_sources, left_admittance, interface_admittance, offsets[1:]):
        (a, b), (c, d) = combined
        (e, f), (g, h) = layer
        # Both local storage and film exchange contribute to the trace admittance.
        d = d + left_shunt
        e = e + shunt
        # Eliminate the two contact temperatures using the heat balance and
        # the contact jump directly, so perfect contact needs no 1/R limit.
        denominator = d + e + r*d*e
        if denominator == 0:
            raise ValueError("interface temperatures are not uniquely determined")
        if wall_temperatures is not None or return_flux_offset:
            source = source-layer_offset[0]
            right_flux_offset = right_flux_offset-left_source
            offset = torch.stack((source-(1+r*e)*right_flux_offset,
                                  (1+r*d)*source-right_flux_offset)) / denominator
            left_flux_offset = left_flux_offset+b*offset[0]
            right_flux_offset = g*offset[1]+layer_offset[1]
            if wall_temperatures is not None:
                # Recover the two contact temperatures when the end drive is known.
                mapping = -torch.stack((torch.stack((c*(1+r*e), f)),
                                        torch.stack((c, f*(1+r*d))))) / denominator
                interfaces.append((mapping, offset))
        left = torch.stack((a - b*c*(1+r*e)/denominator, -b*f/denominator))
        right = torch.stack((-g*c/denominator, h - g*f*(1+r*d)/denominator))
        combined = torch.stack((left, right))
    if return_flux_offset:
        exterior_offset = torch.stack((left_flux_offset, right_flux_offset))
    if wall_temperatures is None:
        return (combined, exterior_offset) if return_flux_offset else combined
    temperatures = []
    ends = drive
    for mapping, offset in reversed(interfaces):
        before, after = mapping @ ends + offset
        temperatures.append(torch.stack((after, ends[1])))
        ends = torch.stack((drive[0], before))
    temperatures.append(ends)
    temperatures = torch.stack(temperatures[::-1])
    fluxes = (torch.stack(matrices) @ temperatures.unsqueeze(-1)).squeeze(-1)
    result = (combined, temperatures, fluxes + offsets)
    return (*result, exterior_offset) if return_flux_offset else result


def solve_heat_wall_admittance_1d(admittance, wall_types, wall_values, *,
                                  flux_offset=(0., 0.), film_coefficients=(0., 0.),
                                  wall_areal_heat_capacity=(0., 0.), angular_frequency=None,
                                  return_external_flux=False, wall_applied_flux=(0., 0.)):
    """Solve two exterior boundary conditions for a harmonic slab or stack.

    Return (wall temperatures in K, inward heat fluxes in W/m²), ordered
    left, right. Supply Y and the optional offset b in q = Y @ T + b.
    The returned q enters the material. Each wall may additionally store
    heat with areal capacity Cw in J/(m² K); supply the common angular
    frequency in rad/s for nonzero capacities. Its external supply is
    q_ext = q + i*omega*Cw*T. Dirichlet data prescribe T, Neumann data
    prescribe q_ext, and Robin data prescribe Ta with q_ext = h*(Ta-T).
    wall_applied_flux adds independent inward heater phasors (W/m²) to the
    prescribed Neumann or Robin supply on either end. Use a nonnegative real
    film coefficient in W/(m² K) on Robin walls only.
    Set return_external_flux=True to append q_ext (W/m²).
    """
    y = torch.as_tensor(admittance, dtype=torch.complex128)
    if y.shape != (2, 2) or not torch.isfinite(y).all():
        raise ValueError("supply a finite 2-by-2 admittance matrix")
    kinds = tuple(wall_types)
    if len(kinds) != 2 or any(k not in ("dirichlet", "neumann", "robin") for k in kinds):
        raise ValueError("supply dirichlet, neumann or robin for each wall")
    values = torch.as_tensor(wall_values, dtype=y.dtype, device=y.device)
    offset = torch.as_tensor(flux_offset, dtype=y.dtype, device=y.device)
    if any(v.shape != (2,) or not torch.isfinite(v).all() for v in (values, offset)):
        raise ValueError("supply two finite wall values and flux offsets")
    if torch.as_tensor(film_coefficients).is_complex():
        raise ValueError("film coefficients must be real")
    film = torch.as_tensor(film_coefficients, dtype=torch.float64, device=y.device)
    robin = torch.tensor([k == "robin" for k in kinds], device=y.device)
    fixed = torch.tensor([k == "dirichlet" for k in kinds], device=y.device)
    if (film.shape != (2,) or not torch.isfinite(film).all() or (film < 0).any()
            or (film[~robin] != 0).any()):
        raise ValueError("supply nonnegative finite films on Robin walls and zero elsewhere")
    if torch.as_tensor(wall_areal_heat_capacity).is_complex():
        raise ValueError("wall areal heat capacities must be real")
    capacity = torch.as_tensor(wall_areal_heat_capacity, dtype=torch.float64, device=y.device)
    if capacity.shape != (2,) or not torch.isfinite(capacity).all() or (capacity < 0).any():
        raise ValueError("supply two nonnegative finite wall areal heat capacities")
    if angular_frequency is None:
        if capacity.any():
            raise ValueError("storing walls require an angular frequency")
        angular_frequency = 0.
    elif not math.isfinite(angular_frequency) or angular_frequency <= 0:
        raise ValueError("angular_frequency must be finite and positive")
    storage = 1j*angular_frequency*capacity
    identity = torch.eye(2, dtype=y.dtype, device=y.device)
    operator = torch.where(fixed[:, None], identity, y+(film+storage).diag())
    heater = _surface_heater(wall_applied_flux, kinds, y)
    imposed = torch.where(robin, film*values, values) + heater
    rhs = torch.where(fixed, values, imposed-offset)
    temperature = torch.linalg.solve(operator, rhs)
    material_flux = y @ temperature + offset
    if return_external_flux:
        return temperature, material_flux, material_flux + storage*temperature
    return temperature, material_flux


def solve_heat_1d_bulk_history(times, ambient_intervals, *, initial_temperature,
                               ncells=16, degree=2, conductivity=1., capacity=1.,
                               heat_loss_coefficient=1., steps_per_interval=8, penalty=20.,
                               inward_flux_intervals=None, film_coefficients=(0.,0.),
                               wall_areal_heat_capacity=(0.,0.), wall_ambient_intervals=None,
                               return_wall_history=False, return_interval_modes=False):
    """Advance c*T_t=k*T_xx+H(x)*(Ta-T) on the unit slab.

    Each interval supplies an independent (start, end) pair of reservoir
    profiles, interpolated linearly in time. inward_flux_intervals optionally
    specifies the two end fluxes in W/m² with shape (interval, start/end,
    left/right). Positive values enter the slab. Both ends default to insulation. A profile may be scalar, cellwise
    or callable at real spatial quadrature points. H is stationary. CPU
    sparse SDIRK2 returns centers, knot modal fields, interval rightward face
    heat and interval reservoir heat into each cell (both in J/m²).
    At either end, external input is the prescribed inward flux plus
    h*(Tfluid-Twall). Supply wall_ambient_intervals with the same shape as
    inward_flux_intervals. A surface capacity in J/(m² K) stores heat at
    the DG endpoint trace, initially the trace of the projected field.
    return_wall_history appends a dict with knot temperatures (temperature,
    K), interval-mean traces (mean_temperature, K) and interval inward
    external heat (external_heat, J/m²), each ordered left then right.
    return_interval_modes appends the SDIRK2 interval-mean DG coefficients
    (interval, cell, mode), after any requested wall history.
    """
    from scipy import sparse
    from scipy.sparse.linalg import splu
    from .sipg_sparse import assemble_heat_1d_sparse

    knots = np.asarray(times,dtype=float)
    if (knots.ndim!=1 or len(knots)<2 or not np.isfinite(knots).all() or np.any(np.diff(knots)<=0)
            or len(ambient_intervals)!=len(knots)-1):
        raise ValueError("supply increasing finite knot times and one ambient pair per interval")
    if ncells<2 or degree<1 or not isinstance(steps_per_interval,int) or steps_per_interval<1:
        raise ValueError("require ncells >= 2, degree >= 1 and positive integer steps_per_interval")
    if not all(math.isfinite(v) and v>0 for v in (conductivity,capacity,penalty)):
        raise ValueError("conductivity, capacity and penalty must be finite and positive")
    def boundary_history(values):
        if values is None:
            return np.zeros((len(knots)-1,2,2))
        data = np.asarray(values)
        if data.shape!=(len(knots)-1,2,2) or np.iscomplexobj(data) or not np.isfinite(data).all():
            raise ValueError("supply finite real boundary pairs at each interval start and end")
        return data
    inward = boundary_history(inward_flux_intervals)
    ambient_wall = boundary_history(wall_ambient_intervals)
    film,cw = map(np.asarray,(film_coefficients,wall_areal_heat_capacity))
    for pair in (film,cw):
        if pair.shape!=(2,) or np.iscomplexobj(pair) or not np.isfinite(pair).all() or (pair<0).any():
            raise ValueError("film and wall capacity require finite nonnegative real pairs")
    h = 1./ncells
    x = (torch.arange(ncells,dtype=torch.float64)+.5)*h
    stiff,mass,ops = assemble_heat_1d_sparse(ncells,degree,penalty,x.device,x.dtype,
                                             wall_types=("neumann","neumann"))
    nodes,weights,phi = ops[:3]
    points = x[:,None]+.5*h*nodes
    exchange,weighted = _bulk_exchange_matrix(_bulk_profile(heat_loss_coefficient,points,x.dtype),weights,phi,h)
    operator = conductivity*stiff
    if exchange is not None:
        operator += exchange
    mass = capacity*mass
    state = _project_initial(x,h,nodes,weights,phi,initial_temperature).detach().numpy().flatten()
    trace = _slab_trace_matrix(ncells,degree,ops[5],ops[6])
    if np.any(film):
        operator += trace.T @ sparse.diags(film) @ trace
    if np.any(cw):
        mass += trace.T @ sparse.diags(cw) @ trace
    shape = (ncells,degree+1)
    history = [torch.from_numpy(state.reshape(shape))]
    face_heat,bulk_heat = [],[]
    interval_modes = []
    surface = trace @ state
    wall_history,mean_wall_history,external_history = [surface],[],[]
    gamma = 1-1/math.sqrt(2.)
    for duration,pair,ends,air in zip(np.diff(knots),ambient_intervals,inward,ambient_wall):
        if len(pair)!=2:
            raise ValueError("each ambient interval requires a start and an end profile")
        begin,end = [((weighted*_bulk_profile(p,points,x.dtype)) @ phi).detach().numpy().ravel() for p in pair]
        # Keep reservoir work separate from the prescribed boundary work.
        net = .5*duration*(begin+end)
        for load,flux_pair in zip((begin,end),ends+film*air):
            local = load.reshape(shape)
            local[0] += flux_pair[0]*ops[5].numpy()
            local[-1] += flux_pair[1]*ops[6].numpy()
        dt = duration/steps_per_interval
        factor = splu((mass+gamma*dt*operator).tocsc())
        integral = np.zeros_like(state)
        for step in range(steps_per_interval):
            load1 = begin+((step+gamma)/steps_per_interval)*(end-begin)
            load2 = begin+((step+1)/steps_per_interval)*(end-begin)
            k1 = factor.solve(load1-operator@state)
            first = state+gamma*dt*k1
            predictor = state+(1-gamma)*dt*k1
            state = predictor+gamma*dt*factor.solve(load2-operator@predictor)
            integral += dt*((1-gamma)*first+gamma*state)
        modes = torch.from_numpy(integral.reshape(shape))
        if return_interval_modes:
            interval_modes.append(modes/duration)
        flux = conductivity*sipg_heat_face_fluxes(modes,penalty)
        next_surface = trace @ state
        integrated_surface = trace @ integral
        external = duration*(ends+film*air).mean(axis=0) - film*integrated_surface
        flux[[0,-1]] = torch.from_numpy((external-cw*(next_surface-surface))*[1.,-1.])
        surface = next_surface
        if return_wall_history:
            wall_history.append(surface)
            mean_wall_history.append(integrated_surface/duration)
            external_history.append(external)
        if exchange is not None:
            net -= exchange@integral
        face_heat.append(flux)
        bulk_heat.append(torch.from_numpy(net.reshape(shape)[:,0]))
        history.append(torch.from_numpy(state.reshape(shape)))
    result = (x,torch.stack(history),torch.stack(face_heat),torch.stack(bulk_heat))
    if return_wall_history:
        result += (dict(temperature=torch.from_numpy(np.stack(wall_history)),
                             mean_temperature=torch.from_numpy(np.stack(mean_wall_history)),
                             external_heat=torch.from_numpy(np.stack(external_history))),)
    if return_interval_modes:
        result += (torch.stack(interval_modes),)
    return result
