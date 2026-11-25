"""Two independently phased Dirichlet ends in finite-interval advection-diffusion."""
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from tensordg.advection_diffusion_1d import sample_solution,solve_transient_advection_diffusion_1d


def measure(n,steps,speed):
    pe,omega = 4.,7.
    d = abs(speed)/pe
    b = speed/(2*d)
    z = np.sqrt(b*b+1j*omega/d)
    left,right = .8+.3j,-.2+.6j
    def profile(x):
        # Removing exp(b*x) from the concentration gives v''=z²*v.
        west,east = left*np.sinh(z*(1-x)),np.exp(-b)*right*np.sinh(z*x)
        scale = np.exp(b*x)/np.sinh(z)
        u = scale*(west+east)
        grad = b*u+scale*z*(-left*np.cosh(z*(1-x))+np.exp(-b)*right*np.cosh(z*x))
        return u,speed*u-d*grad
    end = 2*np.pi/omega
    times = np.linspace(0.,end,steps+1)
    def drive(t):
        phase = np.exp(1j*omega*t)
        return (left*phase).real,(right*phase).real
    def initial(x):
        return torch.from_numpy(profile(x.numpy())[0].real)
    centers,history,_,(west,east) = solve_transient_advection_diffusion_1d(times,initial,
                   ncells=n,degree=3,speed=speed,peclet=pe,left_value=0.,right_value=0.,
                   boundary_value_history=drive,return_face_transport=True)
    x,_ = sample_solution(centers,history[0],12)
    u,_ = profile(x.numpy())
    max_field = 0.
    for t,c in zip(times,history):
        _,uh = sample_solution(centers,c,12)
        max_field = max(max_field,float(abs(uh.numpy()-(u*np.exp(1j*omega*t)).real).max()))
    _,flux = profile(np.array([0.,1.]))
    integral = np.exp(1j*omega*times[:-1])*np.expm1(1j*omega*np.diff(times))/(1j*omega)
    exact = (integral[:,None]*flux).real
    difference = np.column_stack((west[:,0],east[:,-1]))-exact
    return dict(field=max_field,end_mean_flux=float(abs(difference/np.diff(times)[:,None]).max()),
                cumulative_end_transport=float(abs(difference.cumsum(axis=0)).max()),
                cell_balance=float(abs(history[:,:,0].diff(dim=0)/n-west+east).max()))


if __name__=="__main__":
    rows = []
    for speed in (-1.,1.):
        previous = {}
        for steps in (16,32,64,128):
            errors = measure(32,steps,speed)
            orders = {k:float(np.log2(previous[k]/errors[k])) for k in ("field","end_mean_flux","cumulative_end_transport")} if previous else {}
            rows.append(dict(speed=speed,steps=steps,errors=errors,orders=orders,passed=errors["cell_balance"]<1e-10
                             and all(v>0 for v in orders.values()) and (steps<128 or min(orders.values())>1.8)))
            previous = errors
    revision = subprocess.check_output(["git","rev-parse","HEAD"],cwd=Path(__file__).parent,text=True).strip()
    print(json.dumps(dict(commit=revision,runs=rows),indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
