"""Forced and freely decaying Fourier modes on a periodic advection-diffusion ring."""
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from tensordg.advection_diffusion_1d import sample_solution,solve_transient_advection_diffusion_1d


START,STOP,END = .018,.083,.15
SAMPLES = np.array([0.,.01,.025,.04,.075,.095,.12,END])
MODES = ((1,.8+.2j),(2,-.35+.6j))


def measure(n,speed,steps):
    pe = 15.
    times = np.unique(np.round(np.r_[np.linspace(0.,END,steps+1),SAMPLES],14))
    def source(t,x):
        if not START<=t<STOP:
            return 0.
        return 1.2+sum((amp*torch.exp(2j*np.pi*m*x)).real for m,amp in MODES)
    centers,u,_ = solve_transient_advection_diffusion_1d(times,0.,ncells=n,degree=3,speed=speed,
              peclet=pe,periodic=True,source_history=source,source_jump_times=(START,STOP))
    errors = dict(field_l2=0.,mode_max=0.,mass_max=0.)
    weights = torch.from_numpy(np.polynomial.legendre.leggauss(16)[1])/(2*n)
    for t,state in zip(SAMPLES,u[np.searchsorted(times,SAMPLES)]):
        duration = max(0.,min(t,STOP)-START)
        age = max(0.,t-STOP)
        x,uh = sample_solution(centers,state,16)
        x,uh = x.reshape(n,-1),uh.reshape(n,-1)
        exact = torch.full_like(x,1.2*duration)
        for m,amp in MODES:
            k = 2*np.pi*m
            z = k*k*abs(speed)/pe+1j*k*speed
            coeff = amp*(-np.expm1(-duration*z))/z*np.exp(-age*z)
            phase = torch.exp(1j*k*x)
            exact += (coeff*phase).real
            recovered = (2*weights*uh*phase.conj()).sum().item()
            errors["mode_max"] = max(errors["mode_max"],float(abs(recovered-coeff)))
        errors["field_l2"] = max(errors["field_l2"],float(torch.sqrt((weights*(uh-exact).square()).sum())))
        errors["mass_max"] = max(errors["mass_max"],abs(state[:,0].mean().item()-1.2*duration))
    return errors


if __name__=="__main__":
    rows = []
    for speed in (-1.,1.):
        prev = {}
        for steps in (16,32,64,128,256):
            errors = measure(64,speed,steps)
            orders = {k:float(np.log2(prev[k]/errors[k])) for k in ("mode_max","field_l2")} if prev else {}
            passed = errors["mass_max"]<1e-11 and all(v>0 for v in orders.values())
            rows.append(dict(speed=speed,cells=64,nominal_steps=steps,errors=errors,orders=orders,
                             passed=passed and (steps<256 or min(orders.values())>1.8)))
            prev = errors
    revision = subprocess.check_output(["git","rev-parse","HEAD"],cwd=Path(__file__).parent,text=True).strip()
    print(json.dumps(dict(commit=revision,observation_times=SAMPLES.tolist(),runs=rows),indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
