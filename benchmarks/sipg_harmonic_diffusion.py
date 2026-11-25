"""Verify frequency-dependent spatial convergence, phase, cell power and the zero-frequency limit."""
import cmath
import json
import math
from pathlib import Path
import subprocess

import torch

from tensordg import l2_error_2d, poisson_face_powers_2d, solve_harmonic_diffusion_2d, solve_low_frequency_diffusion_2d


def measure(n,degree,ratio):
    kx,ky,capacity,loss = 1.3,2.1,1.7,.8
    stiffness = math.pi**2*(kx+4*ky)+loss
    omega = ratio*stiffness/capacity
    drive = .8+.3j
    def mode(x,y):
        return torch.sin(math.pi*x)*torch.sin(2*math.pi*y)
    transfer = drive/(stiffness+1j*omega*capacity)
    x,y,u = solve_harmonic_diffusion_2d(omega,ncells=n,degree=degree,capacity=capacity,
                 conductivity=(kx,ky),reaction=loss,forcing=lambda x,y:drive*mode(x,y))
    error = l2_error_2d(x,y,u,lambda x,y:transfer*mode(x,y))["relative_l2"]
    average = math.prod(math.sin(t)/t for t in (math.pi/(2*n),math.pi/n))*mode(x[:,None],y)
    modal = ((u[:,:,0,0]*average).sum()/average.square().sum()).item()
    qx,qy = poisson_face_powers_2d(u,conductivity=(kx,ky))
    imbalance = qx.diff(dim=0)+qy.diff(dim=1)-((drive*average)-(loss+1j*omega*capacity)*u[:,:,0,0])/n**2
    return dict(cells=n,degree=degree,omega_rad_s=omega,relative_l2=error,
                relative_amplitude_error=abs(abs(modal/transfer)-1),phase_error_rad=abs(cmath.phase(modal/transfer)),
                cell_imbalance_W=float(abs(imbalance).max()))


def frequency_limit():
    options = dict(ncells=(6,4),degree=2,capacity=lambda x,y:1+x,bc=dict(right="robin"),
                   robin_films={"right":(2.,.3)},forcing=lambda x,y:x+y)
    _,_,steady,lag = solve_low_frequency_diffusion_2d(**options)
    rows = []
    for omega in (1.,.5,.25,.125):
        _,_,field = solve_harmonic_diffusion_2d(omega,**options)
        errors = [float(abs(field.real-steady).max()),float(abs(field.imag/omega-lag).max())]
        orders = [math.log2(a/b) for a,b in zip(rows[-1]["errors"],errors)] if rows else []
        rows.append(dict(omega_rad_s=omega,errors=errors,orders=orders,passed=not orders or min(orders)>1.95))
    return rows


if __name__=="__main__":
    rows = []
    for ratio in (.1,1.,10.):
        for degree in (1,2,3):
            previous = None
            for n in (4,8,16):
                row = measure(n,degree,ratio)
                error = row["relative_l2"]
                order = math.log2(previous/error) if previous else None
                row.update(order=order,passed=row["cell_imbalance_W"]<1e-9 and
                           (n<16 or (order>degree+.65 and row["relative_amplitude_error"]<.02
                                     and row["phase_error_rad"]<.02)))
                rows.append(row)
                previous = error
    limit = frequency_limit()
    revision = subprocess.check_output(["git","rev-parse","HEAD"],cwd=Path(__file__).parent,text=True).strip()
    print(json.dumps(dict(commit=revision,runs=rows,frequency_limit=limit),indent=2,allow_nan=False))
    raise SystemExit(0 if all(r["passed"] for r in rows+limit) else 1)
