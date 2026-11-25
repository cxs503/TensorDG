"""Four rectangular materials with crossing interfaces and a separable analytic field."""
import json
import math
from pathlib import Path
import subprocess

import numpy as np
import torch

from tensordg import l2_error_2d, poisson_face_powers_2d, solve_poisson_2d_bc


def measure(n,degree):
    def material(z,low,high):
        return torch.where(z<.5,torch.full_like(z,low),high)

    def coordinate(z,low,high):
        return z.clamp(max=.5)/low+(z-.5).clamp(min=0)/high

    def angles(x,y):
        return 2*coordinate(x,1.,3.)+.3, 3*coordinate(y,2.,.5)+.2

    def exact(x,y):
        a,b = angles(x,y)
        return torch.sin(a)*torch.cos(b)

    def source(x,y):
        kx,ky = material(x,1.,3.),material(y,2.,.5)
        return (4*ky/kx+9*kx/ky)*exact(x,y)

    center = (torch.arange(n,dtype=torch.float64)+.5)/n
    k = material(center[:,None],1.,3.)*material(center,2.,.5)
    data = dict(conductivity=k,dirichlet_values=dict.fromkeys(("left","right","top","bottom"),exact))
    x,y,u,_ = solve_poisson_2d_bc(n,degree,forcing=source,sparse=True,**data)
    error = l2_error_2d(x,y,u,exact)["absolute_l2"]
    qx,qy = poisson_face_powers_2d(u,**data)
    nodes,weights = (torch.from_numpy(v) for v in np.polynomial.legendre.leggauss(12))
    faces = torch.arange(n+1,dtype=x.dtype)/n
    points = center[:,None]+nodes/(2*n)
    a,b = angles(faces[:,None,None],points)
    ref_x = (-2*material(points,2.,.5)*torch.cos(a)*torch.cos(b))@weights/(2*n)
    a,b = angles(points[:,None,:],faces[:,None])
    ref_y = (3*material(points[:,None,:],1.,3.)*torch.sin(a)*torch.sin(b))@weights/(2*n)
    net = torch.einsum("ijqr,q,r->ij",source(points[:,None,:,None],points[None,:,None,:]),weights,weights)/(4*n*n)
    # Convert face-integrated power to face-averaged heat flux before comparing grids.
    return dict(n=n,degree=degree,l2_error=error,
                flux_error=n*max(float((qx-ref_x).abs().max()),float((qy-ref_y).abs().max())),
                cell_balance=float((net-qx.diff(dim=0)-qy.diff(dim=1)).abs().max()))


if __name__=="__main__":
    rows = []
    for degree in (1,2):
        previous = {}
        for n in (4,8,16,32,64):
            row = measure(n,degree)
            errors = {k:row[k] for k in ("l2_error","flux_error")}
            rates = {k:math.log2(previous[k]/v) for k,v in errors.items()} if previous else {}
            row.update(orders=rates,passed=row["cell_balance"]<1e-9 and
                       (not rates or (rates["l2_error"]>degree+.7 and rates["flux_error"]>0
                       and (n<32 or rates["flux_error"]>degree-.2))))
            rows.append(row)
            previous = errors
    commit = subprocess.check_output(["git","rev-parse","HEAD"],cwd=Path(__file__).parent,text=True).strip()
    print(json.dumps(dict(commit=commit,runs=rows),indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
