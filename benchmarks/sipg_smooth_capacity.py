"""A quadratic temperature mode decaying in a slab with continuously varying heat capacity."""
import json
import math
from pathlib import Path
import subprocess

import numpy as np
import torch

from tensordg import l2_error_2d, poisson_face_powers_2d, solve_transient_diffusion_2d


def measure(steps,theta):
    nx,ny,rate,end = 4,3,3.,.18
    def profile(x,y):
        return 1+x*x+x*y
    def capacity(x,y):
        return 1.+.5*x+2*x*y
    def reaction(x,y):
        return rate*capacity(x,y)+2/profile(x,y)
    # Reaction and the normal derivatives give C*T_t = ΔT - reaction*T
    # for T = exp(-rate*t)*profile, with no volume source.
    fluxes = dict(left=lambda x,y:-y,right=lambda x,y:2+y,
                  bottom=lambda x,y:-x,top=lambda x,y:x)
    bc = dict.fromkeys(fluxes,"neumann")
    time = np.linspace(0.,end,steps+1)
    scale = lambda t: math.exp(-rate*t)
    x,y,u = solve_transient_diffusion_2d(profile,time,ncells=(nx,ny),degree=2,bc=bc,
                neumann_flux=fluxes,load_amplitude=scale,capacity=capacity,reaction=reaction,theta=theta)
    error = l2_error_2d(x,y,u[-1],lambda x,y:scale(end)*profile(x,y))["absolute_l2"]
    # Integrate storage and loss on an independent 12-point quadrature grid.
    z,w = np.polynomial.legendre.leggauss(12)
    phi = np.polynomial.legendre.legvander(z,2)
    xq = x.numpy()[:,None,None,None]+z[None,None,:,None]/(2*nx)
    yq = y.numpy()[None,:,None,None]+z/(2*ny)
    weights = w[:,None]*w/(4*nx*ny)
    fields = np.einsum("txyij,qi,rj->txyqr",u.numpy(),phi,phi)
    stored = (fields*(weights*capacity(xq,yq))).sum(axis=(-1,-2))
    loss = (fields*(weights*reaction(xq,yq))).sum(axis=(-1,-2))
    residual = 0.
    for i,dt in enumerate(np.diff(time)):
        amp = (1-theta)*scale(time[i])+theta*scale(time[i+1])
        applied = {side:lambda x,y,f=f:amp*f(x,y) for side,f in fluxes.items()}
        qx,qy = poisson_face_powers_2d((1-theta)*u[i]+theta*u[i+1],bc=bc,neumann_flux=applied)
        out = (qx.diff(dim=0)+qy.diff(dim=1)).numpy()
        budget = stored[i+1]-stored[i]+dt*(out+(1-theta)*loss[i]+theta*loss[i+1])
        residual = max(residual,float(abs(budget).max()))
    exact_heat = scale(end)*np.sum(weights*capacity(xq,yq)*profile(xq,yq))
    return dict(steps=steps,theta=theta,l2_error=error,heat_error=float(abs(stored[-1].sum()-exact_heat)),
                max_cell_residual=residual)


if __name__=="__main__":
    rows = []
    for theta in (.5,1.):
        previous = {}
        for steps in (8,16,32,64):
            row = measure(steps,theta)
            errors = {key:row[key] for key in ("l2_error","heat_error")}
            orders = {key:math.log2(previous[key]/val) for key,val in errors.items()} if previous else {}
            row.update(orders=orders,passed=all(v>(1.9 if theta==.5 else .9) for v in orders.values())
                       and row["max_cell_residual"]<1e-11)
            rows.append(row)
            previous = errors
    commit = subprocess.check_output(["git","rev-parse","HEAD"],cwd=Path(__file__).parent,text=True).strip()
    print(json.dumps(dict(commit=commit,runs=rows),indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
