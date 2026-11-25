"""Heat exchange between four anisotropic materials in an insulated square."""
import json
import math
from pathlib import Path
import subprocess

import numpy as np
import torch

from tensordg import l2_error_2d, poisson_face_powers_2d, solve_transient_diffusion_2d


def measure(n,degree):
    center = (torch.arange(n,dtype=torch.float64)+.5)/n
    a = torch.where(center[:,None]<.5,1.,3.).double()
    b = torch.where(center<.5,2.,.5).double()
    conductivity = torch.stack((a/b,b/a),dim=-1)
    capacity = 1/(a*b)
    bc = dict.fromkeys(("left","right","bottom","top"),"neumann")
    # dr=dx/a, ds=dy/b turns C*T_t=div(K grad T) into T_t=T_rr+T_ss.
    rmid, smid = .5, .25
    alpha,beta = math.pi/(rmid+.5/3),math.pi/(smid+1.)
    def mode(x,y):
        r = x.clamp(max=.5)+(x-.5).clamp(min=0)/3
        s = y.clamp(max=.5)/2+2*(y-.5).clamp(min=0)
        return torch.cos(alpha*r)*torch.cos(beta*s)
    rate = alpha**2+beta**2
    times = np.linspace(0.,.7/rate,n+1)
    decay = math.exp(-rate*times[-1])
    x,y,u = solve_transient_diffusion_2d(mode,times,ncells=n,degree=degree,bc=bc,capacity=capacity,
                                         conductivity=conductivity)
    error = l2_error_2d(x,y,u[-1],lambda x,y:decay*mode(x,y))["absolute_l2"]
    mid = n//2
    heat_x,heat_y = 0.,0.
    for dt,old,new in zip(np.diff(times),u,u[1:]):
        qx,qy = poisson_face_powers_2d((old+new)/2,bc=bc,conductivity=conductivity)
        heat_x += float(dt)*float(qx[mid,:mid].sum())
        heat_y += float(dt)*float(qy[:mid,mid].sum())
    integral = math.sin(alpha*rmid)*math.sin(beta*smid)*(1-decay)/rate
    exact_x,exact_y = integral*alpha/beta,integral*beta/alpha
    storage = capacity*(u[-1,:,:,0,0]-u[0,:,:,0,0])/n**2
    return dict(n=n,degree=degree,l2_error=error,x_heat_error=abs(heat_x-exact_x),y_heat_error=abs(heat_y-exact_y),
                quadrant_balance=abs(float(storage[:mid,:mid].sum())+heat_x+heat_y),
                total_heat_drift=abs(float(storage.sum())))


if __name__=="__main__":
    rows = []
    for degree in (2,3):
        previous = {}
        for n in (8,16,32):
            row = measure(n,degree)
            errors = {key:value for key,value in row.items() if key.endswith("error")}
            orders = {key:math.log2(previous[key]/val) for key,val in errors.items()} if previous else {}
            # The cubic space resolves the interface gradients sufficiently to
            # check the second-order time integrator on these grids.
            accuracy = (n<32 or (row["l2_error"]<1.1e-5 and max(row["x_heat_error"],row["y_heat_error"])<4e-7))
            row.update(orders=orders,passed=accuracy and all(v>(1.8 if degree==3 else 0.) for v in orders.values())
                       and max(row["quadrant_balance"],row["total_heat_drift"])<1e-10)
            rows.append(row)
            previous = errors
    commit = subprocess.check_output(["git","rev-parse","HEAD"],cwd=Path(__file__).parent,text=True).strip()
    print(json.dumps(dict(commit=commit,runs=rows),indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
