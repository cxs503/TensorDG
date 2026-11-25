"""Injection and removal at two points on a ring, checked by continuum Fourier evolution."""
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from tensordg.advection_diffusion_1d import sample_solution,solve_transient_advection_diffusion_1d


START,STOP,END = .023,.081,.15
OBSERVE = np.array([.015,.04,.07,.09,.12,END])


def measure(n,steps,speed,**options):
    pe = 12.
    times = np.unique(np.round(np.r_[np.linspace(0.,END,steps+1),OBSERVE],14))
    source = torch.zeros(n,dtype=torch.float64)
    source[-1],source[n//4-1] = 1.5,-.6
    centers,u,_ = solve_transient_advection_diffusion_1d(times,0.,ncells=n,degree=3,periodic=True,
        speed=speed,peclet=pe,source_jump_times=(START,STOP),interface_source_history=lambda t:source*float(START<=t<STOP),**options)
    # For positive modes: c'_m+(D*k²+i*a*k)c_m=P0+P1*exp(-i*k/4).
    k = 2*np.pi*np.arange(1,129)
    z = abs(speed)/pe*k*k+1j*speed*k
    drive = 1.5-.6*np.exp(-.25j*k)
    errors = dict(mass=0.,first_four_modes=0.,post_pulse_l2=0.)
    weights = torch.from_numpy(np.polynomial.legendre.leggauss(16)[1])/(2*n)
    for t,state in zip(OBSERVE,u[np.searchsorted(times,OBSERVE)]):
        duration,age = max(0.,min(STOP,t)-START),max(0.,t-STOP)
        mean = .9*duration
        exact = drive*(-np.expm1(-z*duration))/z*np.exp(-z*age)
        x,uh = sample_solution(centers,state,16)
        x,uh = x.reshape(n,-1),uh.reshape(n,-1)
        phase = torch.exp(1j*x[:,:,None]*torch.from_numpy(k))
        recovered = ((weights*uh)[:,:,None]*phase[:,:,:4].conj()).sum(dim=(0,1))
        errors["first_four_modes"] = max(errors["first_four_modes"],float((recovered-torch.from_numpy(exact[:4])).abs().max()))
        errors["mass"] = max(errors["mass"],abs(float(state[:,0].mean())-mean))
        if t>STOP:
            ref = mean+2*(phase*torch.from_numpy(exact)).sum(dim=-1).real
            errors["post_pulse_l2"] = max(errors["post_pulse_l2"],float(torch.sqrt((weights*(uh-ref).square()).sum())))
    return errors


if __name__=="__main__":
    rows = []
    for speed in (-1.,1.):
        for damp in (False,True):
            previous = {}
            for steps in (32,64,128,256):
                err = measure(32,steps,speed,rannacher=damp)
                order = {k:float(np.log2(previous[k]/err[k])) for k in ("first_four_modes","post_pulse_l2")} if previous else {}
                rows.append(dict(rannacher=damp,speed=speed,nominal_steps=steps,errors=err,orders=order,
                                 passed=err["mass"]<1e-10 and all(v>0 for v in order.values()) and
                                 (steps<256 or order["first_four_modes"]>1.8)))
                previous = err
    rev = subprocess.check_output(["git","rev-parse","HEAD"],cwd=Path(__file__).parent,text=True).strip()
    print(json.dumps(dict(commit=rev,observations_s=OBSERVE.tolist(),runs=rows),indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
