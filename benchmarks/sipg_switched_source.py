"""Multiple load switches inside a requested interval of a reacting insulated plate."""
import json
import math
from pathlib import Path
import subprocess

import numpy as np
import torch

from tensordg import solve_transient_diffusion_2d


def measure(n,damped):
    jumps = (.083,.091,.103,.45,.71)
    levels = (.4,4.,-2.,1.3,-.5,.7)
    rate,initial = 2.2,1.
    time = np.linspace(0.,1.,n+1)
    def exact(t):
        # Integrate the scalar ODE exactly between successive switches.
        value,start = initial,0.
        for stop,source in zip((*jumps,1.),levels):
            duration = min(stop,t)-start
            value = source/rate+(value-source/rate)*math.exp(-rate*duration)
            if stop>=t:
                return value
            start = stop
    def input_integral(t):
        return sum(level*max(0.,min(t,right)-left)
                   for left,right,level in zip((0.,*jumps),(*jumps,1.),levels))
    capacity = lambda x,y:1+x+y+x*y
    amplitude = lambda t:levels[np.searchsorted(jumps,t,side="right")]
    _,_,fields,means,load = solve_transient_diffusion_2d(initial,time,ncells=(3,4),degree=2,
        bc=dict.fromkeys(("left","right","top","bottom"),"neumann"),capacity=capacity,
        forcing=capacity,reaction=lambda x,y:rate*capacity(x,y),load_amplitude=amplitude,
        load_jump_times=jumps,rannacher=damped,return_interval_means=True,return_load_means=True)
    numerical = fields[...,0,0]
    ref = torch.tensor([exact(t) for t in time],dtype=fields.dtype)
    higher = fields.flatten(-2)[...,1:].abs().max()
    error = float((numerical-ref[:,None,None]).abs().max())
    dt = torch.from_numpy(np.diff(time))
    exact_input = torch.from_numpy(np.diff([input_integral(t) for t in time]))
    budget = numerical.diff(dim=0)+dt[:,None,None]*(rate*means[...,0,0]-load[:,None,None])
    return dict(n=n,damped=damped,temperature_error=error,higher_mode=float(higher),
                integrated_input_error=float(abs(exact_input-dt*load).max()),
                budget_residual=float(budget.abs().max()))


if __name__=="__main__":
    rows = []
    for damped in (False,True):
        previous = None
        for n in (8,16,32,64,128):
            row = measure(n,damped)
            error = row["temperature_error"]
            order = None if previous is None else math.log2(previous/error)
            row.update(order=order,passed=max(row["integrated_input_error"],row["higher_mode"],row["budget_residual"])<1e-11
                       and (n<128 or error<3e-4))
            rows.append(row)
            previous = error
    commit = subprocess.check_output(["git","rev-parse","HEAD"],cwd=Path(__file__).parent,text=True).strip()
    print(json.dumps(dict(commit=commit,runs=rows),indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
