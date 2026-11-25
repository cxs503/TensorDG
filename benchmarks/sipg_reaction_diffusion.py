"""Screened Poisson convergence and cell heat balances on mixed-boundary rectangles."""
import json
import math
from pathlib import Path
import subprocess

import numpy as np
import torch

from tensordg.poisson_2d_bc import l2_error_2d, poisson_face_powers_2d, solve_poisson_2d_bc


def measure(n, degree):
    pi = math.pi
    def exact(x,y):
        return 1+x+.2*y+torch.cos(pi*x)*torch.cos(pi*y)

    def reaction(x,y):
        return 1+2*x+y*y

    def source(x,y):
        return 2*pi*pi*torch.cos(pi*x)*torch.cos(pi*y)+reaction(x,y)*exact(x,y)

    bc = dict(left="dirichlet",right="robin",top="neumann",bottom="neumann")
    film = lambda x,y: 2+y
    opts = dict(bc=bc,dirichlet_values={"left":exact},neumann_flux={"top":.2,"bottom":-.2},
                robin_films={"right":(film,lambda x,y:exact(x,y)+1/film(x,y))})
    x,y,u,meta = solve_poisson_2d_bc((n,n+2),degree,forcing=source,reaction=reaction,sparse=True,**opts)
    qx,qy = poisson_face_powers_2d(u,**opts)
    out = qx.diff(dim=0)+qy.diff(dim=1)
    # Independent Gauss rule integrates f - reaction*u over each physical cell.
    z,w = np.polynomial.legendre.leggauss(12)
    phi = np.polynomial.legendre.legvander(z,degree)
    numerical = np.einsum("xyij,qi,rj->xyqr",u.numpy(),phi,phi)
    xq = x[:,None,None,None]+torch.from_numpy(z)[None,None,:,None]/(2*n)
    yq = y[None,:,None,None]+torch.from_numpy(z)/(2*len(y))
    net = source(xq,yq).numpy()-reaction(xq,yq).numpy()*numerical
    integrated = np.einsum("xyqr,q,r->xy",net,w,w)/(4*n*len(y))
    return dict(n=n,ny=len(y),degree=degree,dofs=meta["dofs"],
                l2_error=l2_error_2d(x,y,u,exact)["absolute_l2"],
                max_cell_balance=float(abs(integrated-out.numpy()).max()))


if __name__ == "__main__":
    rows = []
    for degree in (1,2,3):
        previous = None
        for n in (4,8,16,32):
            row = measure(n,degree)
            order = None if previous is None else math.log2(previous/row["l2_error"])
            row.update(order=order,passed=(order is None or order>degree+.6) and row["max_cell_balance"]<1e-10)
            rows.append(row)
            previous = row["l2_error"]
    commit = subprocess.check_output(["git","rev-parse","HEAD"],cwd=Path(__file__).parent,text=True).strip()
    print(json.dumps(dict(commit=commit,runs=rows),indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
