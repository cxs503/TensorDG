"""Periodic translating Fourier mode with independently known damping and phase."""
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from tensordg.advection_diffusion_1d import sample_solution,solve_transient_advection_diffusion_1d


def measure(n,degree,speed,steps):
    pe = 18.
    wave = 2*np.pi
    end = .125
    times = np.linspace(0.,end,steps+1)
    t = np.arange(n+1)/n
    edges = torch.from_numpy(t+.25*t*(1-t))
    def initial(x):
        return 1.3+.6*torch.cos(wave*x+.27)
    centers,history,_ = solve_transient_advection_diffusion_1d(times,initial,degree=degree,cell_edges=edges,
                                                               periodic=True,speed=speed,peclet=pe)
    x,u = sample_solution(centers,history[-1],16,cell_edges=edges)
    x,u = x.reshape(n,-1),u.reshape(n,-1)
    weights = edges.diff()[:,None]*torch.from_numpy(np.polynomial.legendre.leggauss(16)[1])/2
    phasor = .6*np.exp(-wave**2*abs(speed)/pe*end+1j*(.27-wave*speed*end))
    exact = 1.3+(phasor*torch.exp(1j*wave*x)).real
    measured = 2*(weights*(u-1.3)*torch.exp(-1j*wave*x)).sum().item()
    return dict(l2=float(torch.sqrt((weights*(u-exact).square()).sum())),
                amplitude=float(abs(abs(measured)-abs(phasor))),phase_rad=float(abs(np.angle(measured/phasor))),
                mass=float(abs(history[:,:,0] @ edges.diff()-1.3).max()))


if __name__=="__main__":
    rows = []
    for speed in (-1.,1.):
        for degree in (1,2,3):
            previous = None
            for n in (4,8,16,32):
                steps = 2*n*n
                errors = measure(n,degree,speed,steps)
                order = float(np.log2(previous/errors["l2"])) if previous else None
                rows.append(dict(study="space",speed=speed,degree=degree,cells=n,steps=steps,
                                 errors=errors,order=order,passed=errors["mass"]<1e-9 and
                                 (n<32 or order>degree+.8)))
                previous = errors["l2"]
        previous = {}
        for steps in (4,8,16,32):
            errors = measure(32,3,speed,steps)
            orders = {k:float(np.log2(previous[k]/errors[k])) for k in ("amplitude","phase_rad")} if previous else {}
            rows.append(dict(study="time",speed=speed,degree=3,cells=32,steps=steps,errors=errors,orders=orders,
                             passed=errors["mass"]<1e-9 and all(v>1.85 for v in orders.values())))
            previous = errors
    rev = subprocess.check_output(["git","rev-parse","HEAD"],cwd=Path(__file__).parent,text=True).strip()
    print(json.dumps(dict(commit=rev,runs=rows),indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
