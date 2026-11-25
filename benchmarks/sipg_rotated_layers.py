"""Sheared thermal eigenmode crossing an interface between two rotated tensors."""
import argparse
import json
import math
from pathlib import Path
import subprocess

import numpy as np
import torch

from tensordg import l2_error_2d, poisson_face_powers_2d, solve_poisson_2d_bc, solve_transient_diffusion_2d


def measure(n,transient):
    # In each layer, dr/dx=1/Kxx and v=y-integral(Kxy/Kxx dx).
    # Equal det(K) makes the transformed mode's time decay rate common.
    det,alpha,beta = 3.,2.,1.5
    rate = alpha**2+det*beta**2
    def material(x):
        k = torch.where(x<.5,torch.ones_like(x),4.)
        tilt = torch.where(x<.5,torch.full_like(x,-.7),.45)
        return k,tilt
    def angles(x,y):
        west,east = x.clamp(max=.5),(x-.5).clamp(min=0)
        return alpha*(west+east/4)+.3,beta*(y+.7*west-.45*east)+.4
    def exact(x,y):
        a,b = angles(x,y)
        return torch.sin(a)*torch.cos(b)
    def power(x,y):
        k,tilt = material(x)
        a,b = angles(x,y)
        qr = -alpha*torch.cos(a)*torch.cos(b)
        return qr,tilt*qr+(det/k)*beta*torch.sin(a)*torch.sin(b)
    x = (torch.arange(n,dtype=torch.float64)+.5)/n
    k,tilt = material(x[:,None].expand(n,n))
    tensor = torch.stack((k,det/k+k*tilt.square(),k*tilt),dim=-1)
    walls = dict.fromkeys(("left","right","bottom","top"),exact)
    opts = dict(conductivity=tensor,dirichlet_values=walls)
    def source(x,y):
        return rate*exact(x,y)/material(x)[0]
    if transient:
        time = np.linspace(0.,.0625,n+1)
        scale = lambda t:math.exp(-rate*t)
        x,y,states = solve_transient_diffusion_2d(exact,time,ncells=n,degree=2,capacity=1/k,
                                                  load_amplitude=scale,**opts)
        u = states[-1]
        amplitude = scale(time[-1])
    else:
        x,y,u,_ = solve_poisson_2d_bc(n,2,forcing=source,sparse=True,**opts)
        amplitude = 1.
    error = l2_error_2d(x,y,u,lambda x,y:amplitude*exact(x,y))["absolute_l2"]
    z,w = (torch.from_numpy(v) for v in np.polynomial.legendre.leggauss(12))
    points = x[:,None]+z/(2*n)
    faces = torch.arange(n+1,dtype=x.dtype)/n
    ref_x = (power(faces[:,None,None],points)[0] @ w)/(2*n)
    ref_y = (power(points[:,None,:],faces[:,None])[1] @ w)/(2*n)
    flux_x = torch.zeros_like(ref_x)
    flux_y = torch.zeros_like(ref_y)
    if transient:
        for i,dt in enumerate(np.diff(time)):
            amp = (scale(time[i])+scale(time[i+1]))/2
            qx,qy = poisson_face_powers_2d((states[i]+states[i+1])/2,conductivity=tensor,
                        dirichlet_values=dict.fromkeys(walls,lambda x,y:amp*exact(x,y)))
            flux_x += dt*qx
            flux_y += dt*qy
        reference_scale = (1-amplitude)/rate
        storage = (states[-1]-states[0])[:,:,0,0]/(n**2*k)
        balance = storage+flux_x.diff(dim=0)+flux_y.diff(dim=1)
    else:
        flux_x,flux_y = poisson_face_powers_2d(u,**opts)
        reference_scale = 1.
        net = torch.einsum("xyqr,q,r->xy",source(points[:,None,:,None],points[None,:,None,:]),w,w)/(4*n**2)
        balance = flux_x.diff(dim=0)+flux_y.diff(dim=1)-net
    return dict(n=n,transient=transient,l2_error=error,
                x_error=n*float(abs(flux_x-reference_scale*ref_x).max()),
                y_error=n*float(abs(flux_y-reference_scale*ref_y).max()),
                cell_balance=float(abs(balance).max()))


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transient",action="store_true")
    args = parser.parse_args()
    rows,previous = [],{}
    for n in (4,8,16,32):
        row = measure(n,args.transient)
        errors = {key:val for key,val in row.items() if key.endswith("error")}
        orders = {key:math.log2(previous[key]/v) for key,v in errors.items()} if previous else {}
        row.update(orders=orders,passed=row["cell_balance"]<1e-9 and all(v>0 for v in orders.values())
                   and (n<32 or min(orders.values())>1.8))
        rows.append(row)
        previous = errors
    commit = subprocess.check_output(["git","rev-parse","HEAD"],cwd=Path(__file__).parent,text=True).strip()
    print(json.dumps(dict(commit=commit,runs=rows),indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
