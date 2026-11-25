"""Steady series resistance and transient diffusion along a layered slab's resistance coordinate."""
import json
import math
from pathlib import Path
import subprocess

import numpy as np
import torch

from tensordg import l2_error_2d, poisson_face_powers_2d, solve_poisson_2d_bc, solve_transient_diffusion_2d


def measure(n,axis):
    k1,k2 = .8,4.
    shape = (n,2) if axis==0 else (2,n)
    centers = (torch.arange(n,dtype=torch.float64)+.5)/n
    k = torch.where(centers<.5,torch.full_like(centers,k1),k2)
    k = k.reshape((n,1) if axis==0 else (1,n)).expand(shape)
    capacity = 1/k
    def resistance(x,y):
        z = x if axis==0 else y
        return z.clamp(max=.5)/k1+(z-.5).clamp(min=0)/k2
    length = .5/k1+.5/k2
    wave = math.pi/length
    def mode(x,y):
        return torch.cos(wave*resistance(x,y))
    west,east = ("left","right") if axis==0 else ("bottom","top")
    insulated = dict.fromkeys(("left","right","bottom","top"),"neumann")
    data = dict(conductivity=k,bc={**insulated,west:"robin",east:"dirichlet"},
                robin_films={west:(2.,3.)},dirichlet_values={east:1.})
    x,y,steady,_ = solve_poisson_2d_bc(shape,2,forcing=lambda x,y:0*x+0*y,sparse=True,**data)
    power = 2/(.5+length)
    steady_error = l2_error_2d(x,y,steady,lambda x,y:1+power*(length-resistance(x,y)))["absolute_l2"]
    powers = poisson_face_powers_2d(steady,**data)[axis].sum(dim=1-axis)
    # C*k=1 transforms the transient equation to u_t=u_rr with dr=dx/k.
    end = .7/wave**2
    times = np.linspace(0.,end,n+1)
    x,y,u = solve_transient_diffusion_2d(mode,times,ncells=shape,degree=2,bc=insulated,capacity=capacity,
                                         conductivity=k)
    decay = math.exp(-wave**2*end)
    error = l2_error_2d(x,y,u[-1],lambda x,y:decay*mode(x,y))["absolute_l2"]
    heat = 0.
    for dt,old,new in zip(np.diff(times),u,u[1:]):
        q = poisson_face_powers_2d((old+new)/2,bc=insulated,conductivity=k)[axis]
        heat += float(dt)*float(q.select(axis,n//2).sum())
    exact_heat = (1-decay)*math.sin(wave*.5/k1)/wave
    stored = capacity*(u[-1,:,:,0,0]-u[0,:,:,0,0])
    return dict(n=n,axis=axis,steady_l2=steady_error,steady_power_error=float((powers-power).abs().max()),
                transient_l2=error,interface_heat_error=abs(heat-exact_heat),
                interface_residual=abs(float(stored.narrow(axis,n//2,n//2).mean())/2-heat),
                insulated_drift=abs(float(stored.mean())))


if __name__=="__main__":
    rows = []
    for axis in (0,1):
        previous = {}
        for n in (8,16,32,64):
            row = measure(n,axis)
            errors = {key:row[key] for key in ("transient_l2","interface_heat_error")}
            rates = {key:math.log2(previous[key]/v) for key,v in errors.items()} if previous else {}
            row.update(orders=rates,passed=all(v>1.8 for v in rates.values())
                       and max(row[key] for key in ("steady_l2","steady_power_error","interface_residual",
                                                    "insulated_drift"))<1e-9)
            rows.append(row)
            previous = errors
    commit = subprocess.check_output(["git","rev-parse","HEAD"],cwd=Path(__file__).parent,text=True).strip()
    print(json.dumps(dict(commit=commit,runs=rows),indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
