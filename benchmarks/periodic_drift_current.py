"""Circulating equilibrium on a periodic graded mesh with independently prescribed mean and current."""
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from tensordg.advection_diffusion_1d import (numerical_face_fluxes,sample_solution,
                                             sample_transport_flux,solve_advection_diffusion_1d)


def measure(n,degree,current):
    pe = 5.
    t = np.arange(n+1)/n
    edges = torch.from_numpy(t+.28*t*(1-t))
    k = 2*np.pi
    def density(x):
        return 1.+.45*torch.sin(k*x+.32)
    def multiplier(x):
        return 1.+.4*torch.cos(k*x)
    def drift(x):
        return (current+multiplier(x)/pe*(.45*k*torch.cos(k*x+.32)))/density(x)
    opts = dict(cell_edges=edges,velocity=drift,diffusivity_profile=multiplier)
    centers,c,_ = solve_advection_diffusion_1d(degree=degree,peclet=pe,mean_value=1.,periodic=True,**opts)
    x,u = sample_solution(centers,c,16,cell_edges=edges)
    _,physical = sample_transport_flux(centers,c,pe,samples_per_cell=16,**opts)
    faces = numerical_face_fluxes(c,pe,periodic=True,**opts)
    weights = .5*edges.diff()[:,None]*torch.from_numpy(np.polynomial.legendre.leggauss(16)[1])
    def l2(z):
        return float(torch.sqrt((weights*z.reshape(n,-1).square()).sum()))
    signs = c.new_tensor([(-1.)**j for j in range(degree+1)])
    seam_error = max(abs(c[0] @ signs-density(edges[0])),abs(c[-1].sum()-density(edges[-1])))
    return dict(density_l2=l2(u-density(x)),physical_flux_l2=l2(physical-current),
                seam_density=float(seam_error),face_current=float((faces-current).abs().max()),
                mass=float(abs(c[:,0] @ edges.diff()-1)),cell_balance=float(faces.diff().abs().max()))


if __name__=="__main__":
    rows = []
    for current in (-.35,.35):
        for degree in (1,2,3):
            previous = {}
            for n in (8,16,32,64):
                errors = measure(n,degree,current)
                orders = {k:float(np.log2(previous[k]/errors[k])) for k in ("density_l2","physical_flux_l2")} if previous else {}
                passed = errors["mass"]<1e-11 and errors["cell_balance"]<1e-9
                if n==64:
                    passed &= orders["density_l2"]>degree+.8 and orders["physical_flux_l2"]>degree-.2 and errors["face_current"]<.001
                rows.append(dict(cells=n,degree=degree,current=current,errors=errors,orders=orders,passed=passed))
                previous = errors
    rev = subprocess.check_output(["git","rev-parse","HEAD"],cwd=Path(__file__).parent,text=True).strip()
    print(json.dumps(dict(commit=rev,runs=rows),indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
