"""Spatial and temporal refinement against exact insulated reaction-diffusion modes."""
import json
import math
from pathlib import Path
import subprocess

import numpy as np
import torch

from tensordg import l2_error_2d, solve_transient_diffusion_2d


def measure(n,degree,steps,theta,uniform):
    end,capacity,reaction = .18,2.3,3.
    bc = dict.fromkeys(("left","right","bottom","top"),"neumann")
    time = np.linspace(0.,end,steps+1)
    def initial(x,y):
        return 1.+0*x+0*y if uniform else torch.cos(math.pi*x)*torch.cos(math.pi*y)
    rate = (reaction+(0. if uniform else 2*math.pi**2))/capacity
    x,y,states = solve_transient_diffusion_2d(initial,time,ncells=n,degree=degree,bc=bc,
                                   capacity=capacity,reaction=reaction,theta=theta)
    error = l2_error_2d(x,y,states[-1],lambda x,y:math.exp(-rate*end)*initial(x,y))
    means = states[:,:,:,0,0].mean(dim=(1,2))
    stored = capacity*means.diff()
    lost = reaction*(end/steps)*((1-theta)*means[:-1]+theta*means[1:])
    return dict(n=n,degree=degree,steps=steps,theta=theta,uniform=uniform,
                l2_error=error["absolute_l2"],heat_balance=float((stored+lost).abs().max()))


if __name__=="__main__":
    rows = []
    for uniform,theta,degree in ((True,.5,1),(True,1.,1),(False,.5,1),(False,.5,2)):
        previous = None
        for size in (4,8,16):
            # Shrink dt quadratically with h for the spatial convergence study.
            row = measure(3 if uniform else size,degree,size if uniform else 4*size**2,theta,uniform)
            order = None if previous is None else math.log2(previous/row["l2_error"])
            target = (2 if theta==.5 else 1) if uniform else degree+1
            row.update(order=order,passed=(order is None or order>target-.2) and row["heat_balance"]<1e-12)
            rows.append(row)
            previous = row["l2_error"]
    commit = subprocess.check_output(["git","rev-parse","HEAD"],cwd=Path(__file__).parent,text=True).strip()
    print(json.dumps(dict(commit=commit,runs=rows),indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
