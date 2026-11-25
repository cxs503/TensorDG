"""Decay of an exact thermal mode with unequal storing Robin walls on a unit slab."""
import json
import math
from pathlib import Path
import subprocess

import numpy as np
import torch
from tensordg import solve_transient_diffusion_2d, poisson_face_powers_2d, l2_error_2d


def measure(n,steps,degree=2):
    k,c,beta,center = 1.4,2.,1.8,.35
    decay = k*beta**2/c
    surface = np.array([.3,.8])
    # Use zero ambient temperatures. Choose the films from each endpoint's
    # exact outward flux and storage rate for cos(beta*(x-center))*exp(-decay*t).
    distance = np.array([center,1-center])
    trace0 = np.cos(beta*distance)
    outward0 = k*beta*np.sin(beta*distance)
    films = outward0/trace0+decay*surface
    options = dict(bc=dict(left="robin",right="robin",top="neumann",bottom="neumann"),
                   robin_films=dict(zip(("left","right"),zip(films,[0.,0.]))),
                   wall_areal_heat_capacity=dict(zip(("left","right"),surface)),conductivity=k)
    times = np.linspace(0.,.25,steps+1)
    def initial(x,y):
        return torch.cos(beta*(x-center))
    x,y,fields,means = solve_transient_diffusion_2d(initial,times,ncells=(n,2),degree=degree,
                      capacity=c,return_interval_means=True,**options)
    dt = np.diff(times)
    signal = np.exp(-decay*times)
    # Integrate the trace in y and evaluate in x at the two ends.
    modes = fields[:,:,:,:,0].mean(2).numpy()
    wall = np.stack([modes[:,0] @ ((-1.)**np.arange(degree+1)),modes[:,-1].sum(1)],axis=1)
    room_heat = dt[:,None]*films*(wall[1:]+wall[:-1])/2
    material_heat = []
    cell_residual = 0.
    for i,step in enumerate(dt):
        diff = fields[i+1]-fields[i]
        qx,qy = poisson_face_powers_2d(means[i],temperature_rate=diff/step,**options)
        material_heat.append(step*np.array([-qx[0].sum().item(),qx[-1].sum().item()]))
        residual = step*(qx.diff(dim=0)+qy.diff(dim=1))+c*diff[:,:,0,0]/(2*n)
        cell_residual = max(cell_residual,float(abs(residual).max()))
    reference_heat = (1-signal[1:,None])*outward0/decay
    error = l2_error_2d(x,y,fields[-1],lambda x,y:signal[-1]*initial(x,y))
    return dict(cells_x=n,steps=steps,degree=degree,relative_l2=error["relative_l2"],
                wall_temperature_error_K=float(abs(wall-signal[:,None]*trace0).max()),
                cumulative_outward_heat_error_J_m2=float(abs(np.cumsum(material_heat,axis=0)-reference_heat).max()),
                cell_balance_J=float(cell_residual),
                wall_balance_J_m2=float(abs(surface*np.diff(wall,axis=0)+room_heat-material_heat).max()))


if __name__=="__main__":
    rows = []
    for study in ("space","time"):
        previous = None
        for level in (1,2,4):
            row = measure(4*level,128,2) if study=="space" else measure(12,4*level,3)
            error = row["relative_l2"]
            order = math.log2(previous/error) if previous else None
            row.update(study=study,order=order,passed=(max(row["cell_balance_J"],row["wall_balance_J_m2"])<1e-10
                       and (order is None or order>(2.7 if study=="space" else 1.9))
                       and (level<4 or row["cumulative_outward_heat_error_J_m2"]<1e-4)))
            rows.append(row)
            previous = error
    rev = subprocess.check_output(["git","rev-parse","HEAD"],cwd=Path(__file__).parent,text=True).strip()
    print(json.dumps(dict(commit=rev,runs=rows),indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
