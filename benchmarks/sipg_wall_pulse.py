"""Sudden heating and cooling of a finite slab, with two off-grid wall temperature jumps."""
import json
import math
from pathlib import Path
import subprocess

import numpy as np
import torch

from tensordg import l2_error_2d, poisson_face_powers_2d, solve_transient_diffusion_2d


def measure(n,damped):
    start,stop,end = .013,.047,.1
    times = np.linspace(0.,end,n//2+1)
    bc = dict(left="dirichlet",right="dirichlet",bottom="neumann",top="neumann")
    def amplitude(t):
        return float(start<=t<stop)
    x,y,field,means,loads = solve_transient_diffusion_2d(0.,times,ncells=(n,2),degree=1,
                bc=bc,dirichlet_values={"left":1.},load_amplitude=amplitude,
                load_jump_times=(start,stop),rannacher=damped,return_interval_means=True,return_load_means=True)
    modes = math.pi*torch.arange(1,97,dtype=x.dtype)
    # The steady affine terms cancel when the delayed unit step is subtracted.
    decay = torch.exp(-modes.square()*(end-stop))-torch.exp(-modes.square()*(end-start))
    def exact(x,y):
        return (torch.sin(x[...,None]*modes) @ (2*decay/modes))
    error = l2_error_2d(x,y,field[-1],exact)["absolute_l2"]
    # Integrate the rightward Fourier flux at each wall analytically.
    sign = (-1.)**torch.arange(1,97)
    heat_left = stop-start+float((2*decay/modes.square()).sum())
    heat_right = stop-start+float((2*sign*decay/modes.square()).sum())
    flux_left,flux_right,balance = 0.,0.,0.
    for i,dt in enumerate(np.diff(times)):
        qx,qy = poisson_face_powers_2d(means[i],bc=bc,dirichlet_values={"left":loads[i]})
        stored = (field[i+1]-field[i])[:,:,0,0]/(2*n)
        balance = max(balance,float(abs(stored+dt*(qx.diff(dim=0)+qy.diff(dim=1))).max()))
        flux_left += float(dt*qx[0].sum())
        flux_right += float(dt*qx[-1].sum())
    qx,_ = poisson_face_powers_2d(field[-1],bc=bc)
    exact_end_flux = -2*float(decay.sum())
    return dict(n=n,damped=damped,final_l2_K=error,final_west_flux_error_W_m2=abs(float(qx[0].sum())-exact_end_flux),
                west_heat_error_J_m2=abs(flux_left-heat_left),east_heat_error_J_m2=abs(flux_right-heat_right),
                cell_balance_J=balance)


if __name__=="__main__":
    rows = []
    for n in (16,32,64,128):
        raw,damped = (measure(n,flag) for flag in (False,True))
        for row in (raw,damped):
            row["passed"] = row["cell_balance_J"]<1e-10
        damped["passed"] &= damped["final_l2_K"]<raw["final_l2_K"] and (
            damped["final_west_flux_error_W_m2"]<raw["final_west_flux_error_W_m2"])
        if n==128:
            damped["passed"] &= (damped["final_l2_K"]<2e-5 and damped["final_west_flux_error_W_m2"]<.02
                                 and max(damped["west_heat_error_J_m2"],damped["east_heat_error_J_m2"])<3e-6)
        rows.extend((raw,damped))
    commit = subprocess.check_output(["git","rev-parse","HEAD"],cwd=Path(__file__).parent,text=True).strip()
    print(json.dumps(dict(commit=commit,runs=rows),indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
