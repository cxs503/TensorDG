"""Periodic wall heating of a reacting slab with an insulated back face."""
import cmath
import json
import math
from pathlib import Path
import subprocess

import numpy as np
import torch

from tensordg import solve_transient_diffusion_2d, poisson_face_powers_2d, l2_error_2d


def measure(n):
    capacity,reaction,omega = 2.3,1.6,5.
    wave = cmath.sqrt(reaction+1j*omega*capacity)
    def phasor(x):
        return torch.cosh(wave*(1-x))/cmath.cosh(wave)
    end = .43
    times = np.linspace(0.,end,n+1)
    bc = dict(left="dirichlet",right="neumann",bottom="neumann",top="neumann")
    x,y,u = solve_transient_diffusion_2d(lambda x,y:phasor(x).real,times,ncells=(n,2),
                degree=2,bc=bc,capacity=capacity,reaction=reaction,dirichlet_values={"left":1.},
                load_amplitude=lambda t:math.cos(omega*t))
    phase = cmath.exp(1j*omega*end)
    error = l2_error_2d(x,y,u[-1],lambda x,y:(phase*phasor(x)).real)
    total_heat = 0.
    balance = 0.
    for i,dt in enumerate(np.diff(times)):
        avg = (u[i]+u[i+1])/2
        wall = (math.cos(omega*times[i])+math.cos(omega*times[i+1]))/2
        qx,qy = poisson_face_powers_2d(avg,bc=bc,dirichlet_values={"left":wall})
        total_heat += dt*float(qx[0].sum())
        stored = capacity*(u[i+1]-u[i])[:,:,0,0]/(n*len(y))
        net_out = qx.diff(dim=0)+qy.diff(dim=1)+reaction*avg[:,:,0,0]/(n*len(y))
        balance = max(balance,float((stored+dt*net_out).abs().max()))
    exact_heat = (wave*cmath.tanh(wave)*(phase-1)/(1j*omega)).real
    return dict(n=n,l2_error=error["absolute_l2"],wall_heat_error=abs(total_heat-exact_heat),max_cell_balance=balance)


if __name__=="__main__":
    rows = []
    previous = {}
    for n in (8,16,32,64):
        row = measure(n)
        errors = {key:row[key] for key in ("l2_error","wall_heat_error")}
        rates = {key:math.log2(previous[key]/val) for key,val in errors.items()} if previous else {}
        row.update(orders=rates,passed=all(v>1.8 for v in rates.values()) and row["max_cell_balance"]<1e-11)
        rows.append(row)
        previous = errors
    commit = subprocess.check_output(["git","rev-parse","HEAD"],cwd=Path(__file__).parent,text=True).strip()
    print(json.dumps(dict(commit=commit,runs=rows),indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
