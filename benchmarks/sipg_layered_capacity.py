"""Insulated two-layer slab with a decaying mode and heat exchange across its midpoint."""
import json
import math
from pathlib import Path
import subprocess

import numpy as np
import torch

from tensordg import l2_error_2d, poisson_face_powers_2d, solve_transient_diffusion_2d


def measure(n,axis):
    pi,end = math.pi,.08
    decay = math.exp(-pi*pi*end)
    times = np.linspace(0.,end,n+1)
    shape = (n,2) if axis==0 else (2,n)
    centers = (torch.arange(n,dtype=torch.float64)+.5)/n
    bc = dict.fromkeys(("left","right","top","bottom"),"neumann")

    def mode(x,y):
        z = x if axis==0 else y
        # These branches match both temperature and its normal derivative at 1/2.
        return torch.where(z<.5,torch.cos(pi*z),torch.cos(3*pi*(1-z))/3)

    capacity = torch.where(centers<.5,1.,9.).reshape((n,1) if axis==0 else (1,n)).expand(shape)
    material = capacity if axis==0 else lambda x,y:torch.where(y<.5,1.,9.)
    x,y,u = solve_transient_diffusion_2d(lambda x,y:2.+mode(x,y),times,ncells=shape,degree=2,
                                         bc=bc,capacity=material)
    error = l2_error_2d(x,y,u[-1],lambda x,y:2.+decay*mode(x,y))["absolute_l2"]
    stored = (capacity*(u[-1,:,:,0,0]-u[0,:,:,0,0])).mean(dim=1-axis)/n
    # The integral of pi*exp(-pi²*t) is the heat transferred into layer two.
    exact_transfer = (1-decay)/pi
    transfer = 0.
    for dt,old,new in zip(np.diff(times),u,u[1:]):
        flux = poisson_face_powers_2d((old+new)/2,bc=bc)[axis]
        transfer += float(dt)*float(flux.select(axis,n//2).sum())
    return dict(n=n,axis=axis,l2_error=error,transfer_error=abs(transfer-exact_transfer),
                layer_heat_error=max(abs(float(stored[:n//2].sum())+exact_transfer),
                                     abs(float(stored[n//2:].sum())-exact_transfer)),
                interface_balance=abs(float(stored[n//2:].sum())-transfer),
                insulated_heat_drift=abs(float(stored.sum())))


if __name__=="__main__":
    rows = []
    for axis in (0,1):
        previous = {}
        for n in (8,16,32,64):
            row = measure(n,axis)
            errors = {key:value for key,value in row.items() if key.endswith("error")}
            orders = {key:math.log2(previous[key]/value) for key,value in errors.items()} if previous else {}
            row.update(orders=orders,passed=all(order>1.8 for order in orders.values())
                       and max(row["interface_balance"],row["insulated_heat_drift"])<1e-11)
            rows.append(row)
            previous = errors
    commit = subprocess.check_output(["git","rev-parse","HEAD"],cwd=Path(__file__).parent,text=True).strip()
    print(json.dumps(dict(commit=commit,runs=rows),indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
