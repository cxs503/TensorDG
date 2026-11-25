"""Implicit time integration of the unit-square mixed-boundary SIPG operator."""

import numpy as np
import torch
from scipy.sparse import bsr_matrix, diags
from scipy.sparse.linalg import splu

from .poisson_2d import _operators
from .poisson_2d_bc import assemble_poisson_2d_bc, _wall_storage


def _capacity_quadrature(grid):
    x,y = grid["x"],grid["y"]
    hx,hy,degree = grid["hx"],grid["hy"],grid["degree"]
    z,w,phi,*_ = _operators(degree,x.device,x.dtype)
    xq = x[:,None,None,None]+hx*z[None,None,:,None]/2
    yq = y[None,:,None,None]+hy*z/2
    return xq,yq,z,w,phi


def _capacity_operator(grid,capacity,quadrature):
    x,y = grid["x"],grid["y"]
    hx,hy = grid["hx"],grid["hy"]
    modes = grid["degree"]+1
    shape = (len(x),len(y),modes,modes)
    samples = None
    if callable(capacity):
        xq,yq,z,w,phi = quadrature
        quadrature_shape = (*shape[:2],len(z),len(z))
    indices = torch.arange(modes,dtype=x.dtype)
    modal_norm = 1/(2*indices+1)
    local_mass = hx*hy*modal_norm[:,None]*modal_norm
    heat_capacity = torch.as_tensor(capacity(xq,yq) if callable(capacity) else capacity,dtype=x.dtype)
    if not torch.isfinite(heat_capacity).all() or (heat_capacity<=0).any():
        raise ValueError("capacity must be finite and positive")
    if heat_capacity.ndim==0 or (not callable(capacity) and heat_capacity.shape==shape[:2]):
        mass = (heat_capacity[...,None,None]*local_mass).expand(shape)
        mass_matrix = diags(mass.reshape(-1).numpy(),format="csc")
        blocks = None
    elif callable(capacity):
        samples = torch.broadcast_to(heat_capacity,quadrature_shape)
        basis = (phi[:,None,:,None]*phi[None,:,None,:]).reshape(len(z)**2,modes**2)
        weighted = (samples*(w[:,None]*w)).flatten(2)
        blocks = (hx*hy/4)*((basis.T*weighted[...,None,:]) @ basis)
        count = len(x)*len(y)
        data = blocks.reshape(count,modes**2,modes**2).numpy()
        mass_matrix = bsr_matrix((data,np.arange(count),np.arange(count+1)),
                                 shape=(count*modes**2,count*modes**2)).tocsc()
    else:
        raise ValueError("capacity must be scalar, an (nx,ny) field, or a function of (x,y)")
    return mass_matrix,local_mass,blocks,samples


def solve_transient_diffusion_2d(initial_temperature, times, *, ncells=16, degree=1,
                                 capacity=1., theta=.5, reaction=0., forcing=None, bc=None,
                                 dirichlet_values=None, neumann_flux=None, robin_films=None,
                                 penalty=20., load_amplitude=None, conductivity=1.,
                                 rannacher=False, return_interval_means=False, load_jump_times=(),
                                 return_load_means=False, source_history=None, wall_areal_heat_capacity=None):
    """Solve C*u_t - div(k grad u) + reaction*u = f with optional time-dependent loads.

    times specifies strictly increasing output times, each interval taking one
    theta step (.5 for Crank–Nicolson, 1 for backward Euler). capacity is a
    positive volumetric heat capacity given as a scalar, an (nx,ny) array of
    cell constants, or a function C(x,y) sampled at element quadrature points. Boundary data and reaction use
    the conventions of assemble_poisson_2d_bc, including its cellwise conductivity. Optional load_amplitude(t)
    scales the assembled source, prescribed temperature and flux data, and
    Robin ambient temperature together. Film coefficients and reaction are
    stationary. wall_areal_heat_capacity maps Robin sides to nonnegative
    J/(m² K) values or functions of (x,y). The wall shares the material trace
    temperature. Its storage also contributes to the initial projection.
    By default the amplitude is one. source_history(t,x,y) adds
    an independently varying volume source in physical units, sampled at
    element quadrature points. Its jumps also belong in load_jump_times. The initial field can be a
    scalar, a function (x,y) projected in capacity-weighted L², or previously computed Legendre
    coefficients (nx,ny,degree+1,degree+1). Returns x, y and a CPU float64
    tensor of modal fields with leading time dimension, including the initial state.
    rannacher=True starts Crank–Nicolson with two backward Euler half steps
    in the first interval and after each declared load jump. load_jump_times
    lists distinct interior times at which load_amplitude changes discontinuously;
    off-grid jumps introduce internal steps. At a jump the callable must give
    the new value; the preceding step samples its immediate left limit.
    return_interval_means=True appends the time-integrator
    weighted modal field on each requested interval for flux/heat integration.
    return_load_means=True appends the corresponding scalar load amplitude
    per requested interval (after the optional modal means).
    """
    times = np.asarray(times,dtype=float)
    if times.ndim != 1 or len(times)<2 or not np.isfinite(times).all() or np.any(np.diff(times)<=0):
        raise ValueError("times must be a finite strictly increasing vector")
    if not .5<=theta<=1:
        raise ValueError("require theta in [0.5,1]")
    if rannacher and theta!=.5:
        raise ValueError("Rannacher startup requires theta=.5")
    jumps = np.asarray(load_jump_times,dtype=float)
    if (jumps.ndim!=1 or not np.isfinite(jumps).all() or len(np.unique(jumps))!=len(jumps)
            or np.any(jumps<=times[0]) or np.any(jumps>=times[-1])):
        raise ValueError("load_jump_times must contain distinct finite interior times")
    segments = np.union1d(times,jumps)
    damp = rannacher & ((segments[:-1]==times[0]) | np.isin(segments[:-1],jumps))
    starts,stops = segments[:-1][damp],segments[1:][damp]
    midpoints = starts+.5*(stops-starts)
    if np.any((midpoints<=starts) | (midpoints>=stops)):
        raise ValueError("damped interval has no representable midpoint")
    internal = np.union1d(segments,midpoints)
    weights = np.where(np.isin(internal[:-1],np.r_[starts,midpoints]),1.,theta)
    if forcing is None:
        forcing = lambda x,y: torch.zeros_like(x+y)
    operator,load,grid,metadata = assemble_poisson_2d_bc(ncells,degree,penalty,bc,forcing,neumann_flux,
                    sparse=True,reaction=reaction,conductivity=conductivity,dirichlet_values=dirichlet_values,robin_films=robin_films)
    x,y = grid["x"],grid["y"]
    hx,hy = grid["hx"],grid["hy"]
    modes = degree+1
    shape = (len(x),len(y),modes,modes)
    quadrature = None
    if callable(capacity) or callable(initial_temperature) or source_history is not None:
        quadrature = _capacity_quadrature(grid)
        xq,yq,z,w,phi = quadrature
        quadrature_shape = (*shape[:2],len(z),len(z))

        def modal_moments(values):
            return (hx*hy/4)*torch.einsum("xyqr,q,r,qi,rj->xyij",values,w,w,phi,phi)

    mass_matrix,local_mass,blocks,samples = _capacity_operator(grid,capacity,quadrature)
    if callable(initial_temperature):
        values = torch.as_tensor(initial_temperature(xq,yq),dtype=x.dtype)
        values = torch.broadcast_to(values,quadrature_shape)
        if blocks is not None:
            values = samples*values
        projected = modal_moments(values)
        initial = (projected/local_mass if blocks is None else
                   torch.linalg.solve(blocks,projected.flatten(2)[...,None]).reshape(shape))
    else:
        initial = torch.as_tensor(initial_temperature,dtype=x.dtype)
        if initial.ndim==0:
            constant = torch.zeros(shape,dtype=x.dtype)
            constant[:,:,0,0] = initial
            initial = constant
    if initial.shape!=shape or not torch.isfinite(initial).all():
        raise ValueError("provide a finite initial temperature or matching modal coefficients")
    if wall_areal_heat_capacity:
        surface,wall_load = _wall_storage(grid,metadata["bc"],wall_areal_heat_capacity,
                         initial_temperature if callable(initial_temperature) else None)
        if wall_load is not None:
            # Bulk projection already represents all volume moments of C*T0.
            initial_load = mass_matrix@initial.reshape(-1).numpy()+wall_load
        mass_matrix = (mass_matrix+surface).tocsc()
        if wall_load is not None:
            initial = torch.from_numpy(splu(mass_matrix).solve(initial_load).reshape(shape))
            if not torch.isfinite(initial).all():
                raise ValueError("initial wall temperature must be finite")
    history = np.empty((len(times),operator.shape[0]))
    state = history[0]
    state[:] = initial.reshape(-1).numpy()
    if return_interval_means:
        modal_means = np.zeros_like(history[:-1])
    if return_load_means:
        load_means = np.zeros(len(times)-1)
    start_values = np.ones(len(internal))
    if load_amplitude is not None:
        def sample(t):
            value = np.asarray(load_amplitude(float(t)),dtype=float)
            if value.ndim or not np.isfinite(value):
                raise ValueError("load_amplitude must return a finite scalar at each time")
            return float(value)
        start_values = np.asarray([sample(t) for t in internal])
    end_values = start_values[1:].copy()
    if load_amplitude is not None:
        for j in np.searchsorted(internal,jumps):
            end_values[j-1] = sample(np.nextafter(internal[j],internal[j-1]))

    def volume_load(t):
        value = source_history(float(t),xq,yq)
        if torch.as_tensor(value).is_complex():
            raise ValueError("source_history must return real values")
        sampled = torch.as_tensor(value,dtype=x.dtype)
        if not torch.isfinite(sampled).all():
            raise ValueError("source_history must return finite values")
        try:
            values = torch.broadcast_to(sampled,quadrature_shape)
        except RuntimeError as exc:
            raise ValueError("source_history must broadcast to the quadrature grid") from exc
        return modal_moments(values).reshape(-1).numpy()

    if source_history is not None:
        source_start = volume_load(internal[0])
        jump_set = set(jumps)
    last_shift = None
    out = 0
    for i,dt in enumerate(np.diff(internal)):
        weight = weights[i]
        shift = weight*dt
        if shift!=last_shift:
            factor = splu((mass_matrix+shift*operator).tocsc())
            last_shift = shift
        amplitude = (1-weight)*start_values[i]+weight*end_values[i]
        rhs = amplitude*load.numpy()-operator@state
        if source_history is not None:
            stop = internal[i+1]
            at_jump = stop in jump_set
            source_end = volume_load(np.nextafter(stop,internal[i]) if at_jump else stop)
            rhs += (1-weight)*source_start+weight*source_end
            source_start = volume_load(stop) if at_jump else source_end
        increment = factor.solve(dt*rhs)
        fraction = dt/(times[out+1]-times[out])
        if return_interval_means:
            modal_means[out] += fraction*(state+weight*increment)
        if return_load_means:
            load_means[out] += fraction*amplitude
        state = state+increment
        if internal[i+1]==times[out+1]:
            out += 1
            history[out] = state
    result = (x,y,torch.from_numpy(history.reshape((len(times),*shape))))
    if return_interval_means:
        result += (torch.from_numpy(modal_means.reshape((len(times)-1,*shape))),)
    if return_load_means:
        result += (torch.from_numpy(load_means),)
    return result


def _frequency_system(ncells,degree,penalty,bc,forcing,neumann_flux,reaction,conductivity,
                      dirichlet_values,robin_films,capacity,complex_load,wall_areal_heat_capacity):
    if forcing is None:
        forcing = lambda x,y: torch.zeros_like(x+y)
    stiffness,load,grid,metadata = assemble_poisson_2d_bc(
        ncells,degree,penalty,bc,forcing,neumann_flux,sparse=True,reaction=reaction,
        conductivity=conductivity,dirichlet_values=dirichlet_values,robin_films=robin_films,complex_load=complex_load)
    quadrature = _capacity_quadrature(grid) if callable(capacity) else None
    mass = _capacity_operator(grid,capacity,quadrature)[0]
    if wall_areal_heat_capacity:
        surface,_ = _wall_storage(grid,metadata["bc"],wall_areal_heat_capacity)
        mass += surface
    return stiffness,mass,load,grid,metadata


def solve_low_frequency_diffusion_2d(*, ncells=16, degree=1, capacity=1., conductivity=1.,
                                     reaction=0., forcing=None, bc=None, dirichlet_values=None,
                                     neumann_flux=None, robin_films=None, penalty=20., wall_areal_heat_capacity=None):
    """Return x, y, U0 and U1 for the SIPG temperature U0 + iω U1 + O(ω²).

    Real, frequency-independent data follow solve_transient_diffusion_2d.
    Both fields are CPU float64 Legendre coefficients. A lossless Neumann
    problem requires zero integrated load and selects zero capacity-weighted
    mean at both orders, including surface storage. wall_areal_heat_capacity maps
    Robin sides to nonnegative scalars or functions of (x,y) in J/(m² K).
    """
    stiffness,mass,load,grid,metadata = _frequency_system(ncells,degree,penalty,bc,forcing,
        neumann_flux,reaction,conductivity,dirichlet_values,robin_films,capacity,False,wall_areal_heat_capacity)
    anchored = metadata["temperature_anchored"]
    stride = (degree+1)**2
    rhs = load.numpy()
    if not anchored and abs(rhs[::stride].sum())>128*np.finfo(float).eps*np.abs(rhs).sum():
        raise ValueError("insulated low-frequency diffusion requires zero net load")
    factor = splu((stiffness if anchored else stiffness[1:,1:]).tocsc())

    if not anchored:
        weights = np.asarray(mass[::stride,:].sum(axis=0)).ravel()

    def solve(vector):
        if anchored:
            return factor.solve(vector)
        coeff = np.zeros_like(vector)
        coeff[1:] = factor.solve(vector[1:])
        coeff[::stride] -= (weights@coeff)/weights[::stride].sum()
        return coeff

    steady = solve(rhs)
    first = solve(-(mass@steady))
    shape = (*metadata["ncells"],degree+1,degree+1)
    return grid["x"],grid["y"],torch.from_numpy(steady.reshape(shape)),torch.from_numpy(first.reshape(shape))



def solve_harmonic_diffusion_2d(angular_frequency, *, ncells=16, degree=1, capacity=1.,
                                conductivity=1., reaction=0., forcing=None, bc=None,
                                dirichlet_values=None, neumann_flux=None, robin_films=None, penalty=20., wall_areal_heat_capacity=None):
    """Solve (K + iω M)U = F for exp(+iωt) and positive angular frequency.

    Volume forcing, prescribed temperatures, inward flux and film ambient
    temperatures may be complex. Materials and film coefficients are real.
    Return x, y and CPU complex128 Legendre coefficients. Face powers can be
    evaluated with poisson_face_powers_2d and the same boundary data. For storing
    Robin faces supply wall_areal_heat_capacity by side and use rate iωU.
    """
    if not np.isfinite(angular_frequency) or angular_frequency<=0:
        raise ValueError("angular frequency must be positive and finite")
    stiffness,mass,load,grid,metadata = _frequency_system(ncells,degree,penalty,bc,forcing,
        neumann_flux,reaction,conductivity,dirichlet_values,robin_films,capacity,True,wall_areal_heat_capacity)
    field = splu((stiffness+1j*angular_frequency*mass).tocsc()).solve(load.numpy())
    shape = (*metadata["ncells"],degree+1,degree+1)
    return grid["x"],grid["y"],torch.from_numpy(field.reshape(shape))
