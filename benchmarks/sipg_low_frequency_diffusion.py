"""Convergence and local heat balance of the steady and first-order SIPG fields."""
import json
import math
from pathlib import Path
import subprocess

import torch
from tensordg import l2_error_2d, poisson_face_powers_2d, solve_low_frequency_diffusion_2d


def measure(n,degree,closed):
    kx,ky,capacity = 1.3,2.1,1.7
    bc = dict.fromkeys(("left","right","top","bottom"),"neumann" if closed else "dirichlet")
    trig = torch.cos if closed else torch.sin

    def mode(x,y):
        return trig(math.pi*x)*trig(2*math.pi*y)

    rate = math.pi**2*(kx+4*ky)
    x,y,steady,lag = solve_low_frequency_diffusion_2d(ncells=n,degree=degree,conductivity=(kx,ky),
        capacity=capacity,bc=bc,forcing=mode)
    balance = []
    errors = []
    # Integral of the separated Fourier mode over each cell.
    average = math.prod(math.sin(t)/t for t in (math.pi/(2*n),math.pi/n))*mode(x[:,None],y)
    for field,scale,load in ((steady,1/rate,average),(lag,-capacity/rate**2,-capacity*steady[:,:,0,0])):
        errors.append(l2_error_2d(x,y,field,lambda x,y:scale*mode(x,y))["relative_l2"])
        qx,qy = poisson_face_powers_2d(field,bc=bc,conductivity=(kx,ky))
        balance.append(float(abs(qx.diff(dim=0)+qy.diff(dim=1)-load/n**2).max()))
    return dict(cells=n,degree=degree,insulated=closed,relative_l2=errors,max_cell_imbalance=balance)


if __name__=="__main__":
    rows = []
    for closed in (False,True):
        for degree in (1,2,3):
            previous = None
            for n in (4,8,16):
                row = measure(n,degree,closed)
                errors = row["relative_l2"]
                orders = [math.log2(a/b) for a,b in zip(previous,errors)] if previous else []
                row.update(orders=orders,passed=max(row["max_cell_imbalance"])<1e-9 and
                           (n<16 or min(orders)>degree+.7))
                rows.append(row)
                previous = errors
    revision = subprocess.check_output(["git","rev-parse","HEAD"],cwd=Path(__file__).parent,text=True).strip()
    print(json.dumps(dict(commit=revision,runs=rows),indent=2,allow_nan=False))
    raise SystemExit(0 if all(r["passed"] for r in rows) else 1)
