"""Graded, three-material DG transport in the cumulative diffusion-resistance coordinate."""
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from tensordg.advection_diffusion_1d import face_fluxes,numerical_face_fluxes,sample_solution,solve_advection_diffusion_1d


def measure(n,degree,speed):
    breaks = np.array([0.,.3,.75,1.])
    materials = np.array([.4,4.,.8])
    lengths = np.diff(breaks)
    param = np.arange(n+1)/n
    local = param+.35*param*(1-param)
    edges = torch.from_numpy(np.r_[np.concatenate([x+h*local[:-1] for x,h in zip(breaks,lengths)]),1.])
    ratio = torch.from_numpy(materials.repeat(n))
    wave = 2*np.pi/np.sum(lengths/materials)
    pe = 5.
    d = abs(speed)/pe
    def phase(x):
        z = sum((x-a).clamp(0.,h)/m for a,h,m in zip(breaks,lengths,materials))
        return wave*z+.37
    def value(x):
        return .7+torch.sin(phase(x))
    def flux(x):
        return speed*value(x)-d*wave*torch.cos(phase(x))
    def source(x):
        arg = phase(x)
        return (speed*wave*torch.cos(arg)+d*wave**2*torch.sin(arg))/ratio[:,None]+(1+x)*value(x)
    opts = dict(cell_edges=edges,diffusivity_ratio=ratio,left_value=value(edges)[0].item(),
                right_value=value(edges)[-1].item())
    centers,coeff,_ = solve_advection_diffusion_1d(degree=degree,peclet=pe,speed=speed,
                    source=source,reaction=lambda x:1+x,**opts)
    x,uh = sample_solution(centers,coeff,samples_per_cell=16,cell_edges=edges)
    x,uh = x.reshape(-1,16),uh.reshape(-1,16)
    weights = .5*edges.diff()[:,None]*torch.from_numpy(np.polynomial.legendre.leggauss(16)[1])
    conservative = numerical_face_fluxes(coeff,pe,speed,**opts)
    one_sided = face_fluxes(coeff,pe,speed,cell_edges=edges,diffusivity_ratio=ratio)
    net = (weights*(source(x)-(1+x)*uh)).sum(dim=1)
    interfaces = [n,2*n]
    return dict(l2=float(torch.sqrt((weights*(uh-value(x))**2).sum())),
                one_sided_flux=float((one_sided-flux(edges[1:])).abs().max()),
                interface_flux=float((conservative[interfaces]-flux(edges)[interfaces]).abs().max()),
                cell_balance=float((conservative.diff()-net).abs().max()))


if __name__=="__main__":
    rows = []
    for speed in (-1.25,1.25):
        for degree in (1,2,3):
            previous = {}
            for n in (4,8,16,32):
                errors = measure(n,degree,speed)
                orders = {k:float(np.log2(previous[k]/errors[k])) for k in ("l2","one_sided_flux")} if previous else {}
                passed = errors["cell_balance"]<1e-9 and all(v>0 for v in orders.values())
                if n==32:
                    passed &= orders["l2"]>degree+.8 and orders["one_sided_flux"]>degree-.2 and errors["interface_flux"]<.01
                rows.append(dict(speed=speed,degree=degree,cells_per_material=n,errors=errors,orders=orders,passed=passed))
                previous = errors
    rev = subprocess.check_output(["git","rev-parse","HEAD"],cwd=Path(__file__).parent,text=True).strip()
    print(json.dumps(dict(commit=rev,runs=rows),indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
