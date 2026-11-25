"""Interval-averaged DG face currents against the exact integral of a traveling Fourier mode."""
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from tensordg.advection_diffusion_1d import solve_transient_advection_diffusion_1d


def measure(n,steps,speed,damped):
    pe,k = 18.,2*np.pi
    d = abs(speed)/pe
    times = np.linspace(0.,.16,steps+1)
    xi = np.arange(n+1)/n
    edges = xi+.25*xi*(1-xi)
    def initial(x):
        return .3+.8*torch.cos(k*x+.23)
    _,states,_,(west,east) = solve_transient_advection_diffusion_1d(times,initial,degree=3,
                cell_edges=edges,periodic=True,speed=speed,peclet=pe,rannacher=damped,return_face_transport=True)
    dt = np.diff(times)
    z = d*k*k+1j*speed*k
    # Integrate J=a*u-D*u_x over each interval; expm1 remains accurate for small dt.
    integrated_mode = np.exp(-z*times[:-1])*(-np.expm1(-z*dt))/z
    phasor = .8*(speed-1j*d*k)*np.exp(1j*(k*edges+.23))
    exact = .3*speed*dt[:,None]+(integrated_mode[:,None]*phasor).real
    error = np.maximum(abs(west.numpy()-exact[:,:-1]).max(axis=1),
                       abs(east.numpy()-exact[:,1:]).max(axis=1))/dt
    balance = states[:,:,0].diff(dim=0)*torch.from_numpy(np.diff(edges))-west+east
    return dict(mean_current_max=float(error.max()),late_mean_current_max=float(error[steps//2:].max()),
                cumulative_transport_max=float(abs((east.numpy()-exact[:,1:]).cumsum(axis=0)).max()),
                local_balance=float(balance.abs().max()))


if __name__=="__main__":
    rows = []
    for speed in (-1.,1.):
        for damped in (False,True):
            previous = {}
            for steps in (8,16,32,64):
                errors = measure(32,steps,speed,damped)
                orders = {k:float(np.log2(previous[k]/errors[k])) for k in errors if k!="local_balance"} if previous else {}
                passed = errors["local_balance"]<1e-11 and all(v>0 for v in orders.values())
                if steps==64:
                    passed &= orders["mean_current_max"]>(.85 if damped else 1.8)
                    passed &= min(orders[k] for k in ("late_mean_current_max","cumulative_transport_max"))>1.8
                rows.append(dict(speed=speed,rannacher=damped,steps=steps,errors=errors,orders=orders,passed=passed))
                previous = errors
    revision = subprocess.check_output(["git","rev-parse","HEAD"],cwd=Path(__file__).parent,text=True).strip()
    print(json.dumps(dict(commit=revision,runs=rows),indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
