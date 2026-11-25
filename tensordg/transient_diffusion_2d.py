"""Implicit time integration of the unit-square mixed-boundary SIPG operator."""
import math

import numpy as np
import torch
from scipy.sparse import diags
from scipy.sparse.linalg import splu

from .poisson_2d import _operators
from .poisson_2d_bc import assemble_poisson_2d_bc


def solve_transient_diffusion_2d(initial_temperature, times, *, ncells=16, degree=1,
                                 capacity=1., theta=.5, reaction=0., forcing=None, bc=None,
                                 dirichlet_values=None, neumann_flux=None, robin_films=None,
                                 penalty=20.):
    """Solve C*u_t - Δu + reaction*u = f with stationary loads and unit conductivity.

    times specifies strictly increasing output times, each interval taking one
    theta step (.5 for Crank–Nicolson, 1 for backward Euler). capacity is a
    positive uniform volumetric heat capacity. Boundary data and reaction use
    the conventions of assemble_poisson_2d_bc. The initial field can be a
    scalar, a function (x,y) projected in L², or previously computed Legendre
    coefficients (nx,ny,degree+1,degree+1). Returns x, y and a CPU float64
    tensor of modal fields with leading time dimension, including the initial state.
    """
    times = np.asarray(times,dtype=float)
    if times.ndim != 1 or len(times)<2 or not np.isfinite(times).all() or np.any(np.diff(times)<=0):
        raise ValueError("times must be a finite strictly increasing vector")
    if not math.isfinite(capacity) or capacity<=0 or not .5<=theta<=1:
        raise ValueError("require finite positive capacity and theta in [0.5,1]")
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
    if callable(initial_temperature):
        z,w,phi,*_ = _operators(degree,x.device,x.dtype)
        xq = x[:,None,None,None]+hx*z[None,None,:,None]/2
        yq = y[None,:,None,None]+hy*z/2
        values = torch.as_tensor(initial_temperature(xq,yq),dtype=x.dtype)
        values = torch.broadcast_to(values,(*shape[:2],len(z),len(z)))
        projected = torch.einsum("xyqr,q,r,qi,rj->xyij",values,w,w,phi,phi)
        initial = (hx*hy/4)*projected/local_mass
    else:
        initial = torch.as_tensor(initial_temperature,dtype=x.dtype)
        if initial.ndim==0:
            constant = torch.zeros(shape,dtype=x.dtype)
            constant[:,:,0,0] = initial
            initial = constant
    if initial.shape!=shape or not torch.isfinite(initial).all():
        raise ValueError("provide a finite initial temperature or matching modal coefficients")
    mass = (capacity*local_mass).expand(shape).reshape(-1).numpy()
    mass_matrix = diags(mass,format="csc")
    history = np.empty((len(times),mass.size))
    history[0] = initial.reshape(-1).numpy()
    last_dt = None
    for i,dt in enumerate(np.diff(times)):
        if dt!=last_dt:
            factor = splu((mass_matrix+theta*dt*operator).tocsc())
            last_dt = dt
        state = history[i]
        # Solve for the increment to retain stationary states to roundoff.
        history[i+1] = state+factor.solve(dt*(load.numpy()-operator@state))
    return x,y,torch.from_numpy(history.reshape((len(times),*shape)))
