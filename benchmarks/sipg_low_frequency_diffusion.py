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


def unequal_films(n,degree):
    # With unit conductivity and steady temperature the lag has curvature 1+x.
    # Its endpoint derivatives equal hL*T1(0) and -hR*T1(1).
    left,right = 2.,5.
    slope = -(2/3+1.5/right)/(1+1/left+1/right)
    def exact(x,y):
        return x**3/6+x*x/2+slope*x+slope/left

    bc = dict(left="robin",right="robin",top="neumann",bottom="neumann")
    films = {"left":(left,1.),"right":(right,1.)}
    x,y,steady,lag = solve_low_frequency_diffusion_2d(ncells=(n,3),degree=degree,
                    capacity=lambda x,y:1+x,bc=bc,robin_films=films)
    error = l2_error_2d(x,y,lag,exact)["relative_l2"]
    qx,qy = poisson_face_powers_2d(lag,bc=bc,robin_films={"left":(left,0.),"right":(right,0.)})
    faces = torch.arange(n+1,dtype=x.dtype)/n
    reference = -(faces+.5*faces**2+slope)
    flux_error = float(abs(qx*len(y)-reference[:,None]).max())
    balance = qx.diff(dim=0)+qy.diff(dim=1)+(1+x[:,None])/(n*len(y))
    steady_error = l2_error_2d(x,y,steady,lambda x,y:torch.ones_like(x+y))["absolute_l2"]
    return dict(cells_x=n,degree=degree,relative_l2=error,steady_error_K=steady_error,
                max_flux_lag_error=flux_error,max_cell_imbalance=float(abs(balance).max()))


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
    films = []
    for degree in (1,2,3):
        previous = None
        for n in (4,8,16):
            row = unequal_films(n,degree)
            error = row["relative_l2"]
            order = math.log2(previous/error) if previous and degree<3 else None
            passed = (row["max_flux_lag_error"]<1e-9 and row["max_cell_imbalance"]<1e-10
                      and row["steady_error_K"]<1e-10 and (degree<3 or error<1e-9)
                      and (degree==3 or n<16 or order>degree+.7))
            row.update(order=order,passed=passed)
            films.append(row)
            previous = error
    revision = subprocess.check_output(["git","rev-parse","HEAD"],cwd=Path(__file__).parent,text=True).strip()
    print(json.dumps(dict(commit=revision,runs=rows,unequal_films=films),indent=2,allow_nan=False))
    raise SystemExit(0 if all(r["passed"] for r in rows+films) else 1)
