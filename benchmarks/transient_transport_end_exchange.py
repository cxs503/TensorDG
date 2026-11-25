"""Time-dependent throughflow and reservoir exchange for a quadratic concentration field."""
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from tensordg.advection_diffusion_1d import solve_transient_advection_diffusion_1d,sample_solution


def measure(steps,speed,exchange):
    n,pe,omega = 12,5.,9.
    d = abs(speed)/pe
    edges = np.linspace(0.,1.,n+1)
    times = np.linspace(0.,.4,steps+1)
    def profile(x):
        return .7+x-1.4*x*x
    def current(x):
        return speed*profile(x)-d*(1-2.8*x)
    def signal(t):
        return np.cos(omega*t)+.4*np.sin(omega*t)
    def source(t,x):
        slope = omega*(-np.sin(omega*t)+.4*np.cos(omega*t))
        return slope*profile(x)+signal(t)*(speed*(1-2.8*x)+2.8*d)
    k = np.array(exchange)
    ambient = np.array([.3,-.2])
    ends = np.array([0.,1.])
    normal = np.array([-1.,1.])
    def loads(t):
        return current(ends)*signal(t)-normal*k*(profile(ends)*signal(t)-ambient)
    centers,history,_,(west,east) = solve_transient_advection_diffusion_1d(
        times,profile,ncells=n,degree=2,peclet=pe,speed=speed,boundary_exchange=tuple(zip(k,ambient)),
        source_history=source,total_flux_history=loads,return_face_transport=True)
    integral = np.diff((np.sin(omega*times)-.4*np.cos(omega*times))/omega)
    exact = integral[:,None]*current(edges)
    error = np.maximum(abs(west.numpy()-exact[:,:-1]).max(axis=1),abs(east.numpy()-exact[:,1:]).max(axis=1))
    field_error = 0.
    for t,c in zip(times,history):
        x,u = sample_solution(centers,c)
        field_error = max(field_error,float((u-signal(t)*profile(x)).abs().max()))
    end_error = np.column_stack((west[:,0].numpy()-exact[:,0],east[:,-1].numpy()-exact[:,-1]))
    return dict(concentration=field_error,mean_current=float((error/np.diff(times)).max()),
                cumulative_end_transport=float(abs(end_error.cumsum(axis=0)).max()))


if __name__=="__main__":
    rows = []
    for speed in (-1.,1.):
        for exchange in ((0.,1.7),(1.3,0.),(1.3,1.7)):
            previous = {}
            for steps in (8,16,32,64):
                errors = measure(steps,speed,exchange)
                orders = {k:float(np.log2(previous[k]/v)) for k,v in errors.items()} if previous else {}
                rows.append(dict(speed=speed,exchange=exchange,steps=steps,errors=errors,orders=orders,
                                 passed=all(v>0 for v in orders.values()) and (steps<64 or min(orders.values())>1.85)))
                previous = errors
    rev = subprocess.check_output(["git","rev-parse","HEAD"],cwd=Path(__file__).parent,text=True).strip()
    print(json.dumps(dict(commit=rev,runs=rows),indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
