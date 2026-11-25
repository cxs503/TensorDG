"""Reflecting drift-diffusion equilibrium with analytically normalized Gaussian density."""
import json
import math
from pathlib import Path
import subprocess

import numpy as np
import torch

from tensordg.advection_diffusion_1d import (numerical_face_fluxes,sample_solution,
                                             sample_transport_flux,solve_advection_diffusion_1d)


def measure(n,degree,center):
    beta,pe,mass = 35.,20.,1.4
    edges = np.linspace(0.,1.,n+1)
    if center>1:
        # Refine towards the wall where the truncated density peaks.
        edges = 1-np.expm1(2*edges[::-1])/np.expm1(2.)
    def drift(x):
        return -2*beta/pe*(x-center)
    opts = dict(cell_edges=edges,velocity=drift)
    centers,c,_ = solve_advection_diffusion_1d(degree=degree,peclet=pe,total_flux=(0.,0.),mean_value=mass,**opts)
    x,u = sample_solution(centers,c,16,cell_edges=edges)
    _,j = sample_transport_flux(centers,c,pe,samples_per_cell=16,**opts)
    flux = numerical_face_fluxes(c,pe,total_flux=(0.,0.),**opts)
    norm = math.sqrt(math.pi)/(2*math.sqrt(beta))*(math.erf(math.sqrt(beta)*(1-center))+math.erf(math.sqrt(beta)*center))
    exact = mass*torch.exp(-beta*(x-center)**2)/norm
    weights = torch.from_numpy(np.diff(edges)[:,None]*np.polynomial.legendre.leggauss(16)[1]/2)
    def l2(z):
        return float(torch.sqrt((weights*z.reshape(n,-1).square()).sum()))
    return dict(relative_density_l2=l2(u-exact)/l2(exact),physical_flux_l2=l2(j),
                conservative_flux_max=float(flux.abs().max()),mass_error=float(abs(c[:,0] @ torch.from_numpy(np.diff(edges))-mass)))


if __name__=="__main__":
    rows = []
    for center in (.37,1.15):
        for degree in (1,2,3):
            previous = {}
            for n in (8,16,32,64):
                errors = measure(n,degree,center)
                orders = {k:float(np.log2(previous[k]/errors[k])) for k in ("relative_density_l2","physical_flux_l2")} if previous else {}
                passed = errors["mass_error"]<1e-11 and errors["conservative_flux_max"]<1e-8
                if n==64:
                    passed &= orders["relative_density_l2"]>degree+.8 and orders["physical_flux_l2"]>degree-.2
                rows.append(dict(center=center,cells=n,degree=degree,errors=errors,orders=orders,passed=passed))
                previous = errors
    revision = subprocess.check_output(["git","rev-parse","HEAD"],cwd=Path(__file__).parent,text=True).strip()
    print(json.dumps(dict(commit=revision,runs=rows),indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
