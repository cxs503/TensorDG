"""Pure-flux patch test along the weak axis of a highly anisotropic rotated material."""
import json
import math
from pathlib import Path
import subprocess

import torch

from tensordg import l2_error_2d, poisson_face_powers_2d, solve_poisson_2d_bc


def measure(ratio,angle):
    nx,ny = 6,5
    a,b = math.cos(math.radians(angle)),math.sin(math.radians(angle))
    # (a,b) conducts with strength one; its perpendicular conducts at ratio.
    tensor = (a*a+ratio*b*b,b*b+ratio*a*a,(1-ratio)*a*b)
    def z(x,y):
        return a*x+b*y
    def exact(x,y):
        return z(x,y)**2
    sides = dict(left=lambda x,y:-2*a*z(x,y),right=lambda x,y:2*a*z(x,y),
                 bottom=lambda x,y:-2*b*z(x,y),top=lambda x,y:2*b*z(x,y))
    opts = dict(conductivity=tensor,bc=dict.fromkeys(sides,"neumann"),neumann_flux=sides)
    x,y,u,meta = solve_poisson_2d_bc((nx,ny),2,forcing=lambda x,y:-2+0*x+0*y,sparse=True,**opts)
    qx,qy = poisson_face_powers_2d(u,**opts)
    faces_x = torch.arange(nx+1,dtype=x.dtype)[:,None]/nx
    faces_y = torch.arange(ny+1,dtype=y.dtype)/ny
    err = l2_error_2d(x,y,u,exact,remove_constant=True)["absolute_l2"]
    flux_error = max(float((qx*ny+2*a*z(faces_x,y)).abs().max()),
                     float((qy*nx+2*b*z(x[:,None],faces_y)).abs().max()))
    balance = float((qx.diff(dim=0)+qy.diff(dim=1)+2/(nx*ny)).abs().max())
    return dict(ratio=ratio,angle_deg=angle,l2_error=err,mean_flux_error=flux_error,
                max_cell_residual=balance,nullspace_residual=meta["nullspace_residual"],
                passed=err<1e-7 and flux_error<1e-7 and balance<1e-8)


if __name__=="__main__":
    rows = [measure(ratio,angle) for ratio in (1.,100.,10000.) for angle in (-35.,20.,65.)]
    commit = subprocess.check_output(["git","rev-parse","HEAD"],cwd=Path(__file__).parent,text=True).strip()
    print(json.dumps(dict(commit=commit,runs=rows),indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
