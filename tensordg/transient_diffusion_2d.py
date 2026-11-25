"""Implicit time integration of the unit-square mixed-boundary SIPG operator."""

import numpy as np
import torch
from scipy.sparse import bsr_matrix, diags
from scipy.sparse.linalg import splu

from .poisson_2d import _operators
from .poisson_2d_bc import assemble_poisson_2d_bc


def solve_transient_diffusion_2d(initial_temperature, times, *, ncells=16, degree=1,
                                 capacity=1., theta=.5, reaction=0., forcing=None, bc=None,
                                 dirichlet_values=None, neumann_flux=None, robin_films=None,
                                 penalty=20., load_amplitude=None):
    """Solve C*u_t - Δu + reaction*u = f with separable loads and unit conductivity.

    times specifies strictly increasing output times, each interval taking one
    theta step (.5 for Crank–Nicolson, 1 for backward Euler). capacity is a
    positive volumetric heat capacity given as a scalar, an (nx,ny) array of
    cell constants, or a function C(x,y) sampled at element quadrature points. Boundary data and reaction use
    the conventions of assemble_poisson_2d_bc. Optional load_amplitude(t)
    scales the assembled source, prescribed temperature and flux data, and
    Robin ambient temperature together. Film coefficients and reaction are
    stationary. By default the amplitude is one. The initial field can be a
    scalar, a function (x,y) projected in capacity-weighted L², or previously computed Legendre
    coefficients (nx,ny,degree+1,degree+1). Returns x, y and a CPU float64
    tensor of modal fields with leading time dimension, including the initial state.
    """
    times = np.asarray(times,dtype=float)
    if times.ndim != 1 or len(times)<2 or not np.isfinite(times).all() or np.any(np.diff(times)<=0):
        raise ValueError("times must be a finite strictly increasing vector")
    if not .5<=theta<=1:
        raise ValueError("require theta in [0.5,1]")
    if forcing is None:
        forcing = lambda x,y: torch.zeros_like(x+y)
    operator,load,grid,_ = assemble_poisson_2d_bc(ncells,degree,penalty,bc,forcing,neumann_flux,
                    sparse=True,reaction=reaction,dirichlet_values=dirichlet_values,robin_films=robin_films)
    x,y = grid["x"],grid["y"]
    hx,hy = grid["hx"],grid["hy"]
    modes = degree+1
    shape = (len(x),len(y),modes,modes)
    indices = torch.arange(modes,dtype=x.dtype)
    modal_norm = 1/(2*indices+1)
    local_mass = hx*hy*modal_norm[:,None]*modal_norm
    if callable(capacity) or callable(initial_temperature):
        z,w,phi,*_ = _operators(degree,x.device,x.dtype)
        xq = x[:,None,None,None]+hx*z[None,None,:,None]/2
        yq = y[None,:,None,None]+hy*z/2
        quadrature_shape = (*shape[:2],len(z),len(z))
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
                                 shape=operator.shape).tocsc()
    else:
        raise ValueError("capacity must be scalar, an (nx,ny) field, or a function of (x,y)")
    if callable(initial_temperature):
        values = torch.as_tensor(initial_temperature(xq,yq),dtype=x.dtype)
        values = torch.broadcast_to(values,quadrature_shape)
        if blocks is not None:
            values = samples*values
        projected = (hx*hy/4)*torch.einsum("xyqr,q,r,qi,rj->xyij",values,w,w,phi,phi)
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
    history = np.empty((len(times),operator.shape[0]))
    history[0] = initial.reshape(-1).numpy()
    amplitudes = np.ones(len(times))
    if load_amplitude is not None:
        amplitudes = np.asarray([load_amplitude(float(t)) for t in times],dtype=float)
        if amplitudes.shape!=times.shape or not np.isfinite(amplitudes).all():
            raise ValueError("load_amplitude must return a finite scalar at each time")
    last_dt = None
    for i,dt in enumerate(np.diff(times)):
        if dt!=last_dt:
            factor = splu((mass_matrix+theta*dt*operator).tocsc())
            last_dt = dt
        state = history[i]
        # Solve for the increment to retain stationary states to roundoff.
        amplitude = (1-theta)*amplitudes[i]+theta*amplitudes[i+1]
        history[i+1] = state+factor.solve(dt*(amplitude*load.numpy()-operator@state))
    return x,y,torch.from_numpy(history.reshape((len(times),*shape)))
