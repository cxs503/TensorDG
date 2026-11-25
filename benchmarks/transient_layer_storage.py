"""Compare two-material transient transport with a piecewise analytic decay mode."""
import json
import math
from pathlib import Path
import subprocess

import numpy as np
import torch

from tensordg.advection_diffusion_1d import sample_solution, solve_transient_advection_diffusion_1d


def measure(n, speed):
    interface = .4
    capacity = (1.2, 3.4)
    diffusivity = (.25, .6)
    decay = 4.
    edges = np.r_[np.linspace(0., interface, n+1), np.linspace(interface, 1., n+1)[1:]]
    times = np.linspace(0., .25, 2*n+1)

    def mode(x, layer):
        # Match f and J=speed*f-D*f' at the interface for both materials.
        d = diffusivity[layer]
        b = speed/(2*d)
        w = math.sqrt(decay*capacity[layer]/d-b*b)
        y = x-interface
        a = .8
        h = (b*a-.35/d)/w
        g = a*torch.cos(w*y)+h*torch.sin(w*y)
        dg = w*(-a*torch.sin(w*y)+h*torch.cos(w*y))
        f = torch.exp(b*y)*g
        return f, speed*f-d*torch.exp(b*y)*(b*g+dg)

    def profile(x):
        return torch.where(x<interface, mode(x, 0)[0], mode(x, 1)[0])

    ends = profile(torch.tensor([0.,1.], dtype=torch.float64))
    centers, states, _, (west,east) = solve_transient_advection_diffusion_1d(
        times, profile, cell_edges=edges, degree=3, speed=speed, peclet=4.,
        storage=np.repeat(capacity,n), diffusivity_ratio=np.repeat(np.array(diffusivity)*4,n),
        left_value=0., right_value=0., boundary_value_history=lambda t: ends*math.exp(-decay*t),
        return_face_transport=True)
    field_error = 0.
    for t,state in zip(times,states):
        x,u = sample_solution(centers,state,cell_edges=edges)
        field_error = max(field_error,float((u-profile(x)*math.exp(-decay*t)).abs().max()))
    width_capacity = np.diff(edges)*np.repeat(capacity,n)
    stored = (states[:,:,0].numpy()*width_capacity).reshape(len(times),2,n).sum(axis=2)
    x = torch.as_tensor(edges[[0,n,-1]])
    currents = np.array([mode(x[0],0)[1].item(), .35, mode(x[-1],1)[1].item()])
    exact_stored = np.exp(-decay*times[:,None])*np.diff(currents)/decay
    dt = np.diff(times)
    mean_factor = np.exp(-decay*times[:-1])*(-np.expm1(-decay*dt))/(decay*dt)
    mean_flux = np.column_stack((west[:,0],east[:,n-1],west[:,n],east[:,-1]))/dt[:,None]
    balance = np.diff(stored,axis=0)-(west[:,[0,n]]-east[:,[n-1,-1]]).numpy()
    return dict(field=field_error, stored=float(abs(stored-exact_stored).max()),
                mean_flux=float(abs(mean_flux-mean_factor[:,None]*currents[[0,1,1,2]]).max()),
                balance=float(abs(balance).max()))


if __name__ == "__main__":
    rows = []
    for speed in (-1.,1.):
        previous = {}
        for n in (4,8,16,32):
            errors = measure(n,speed)
            orders = {k:math.log2(previous[k]/errors[k]) for k in ("field","stored","mean_flux") if previous}
            passed = errors["balance"] < 1.e-10 and (n!=32 or min(orders.values())>1.8)
            rows.append(dict(cells_per_layer=n,speed=speed,errors=errors,orders=orders,passed=passed))
            previous = errors
    rev = subprocess.check_output(["git","rev-parse","HEAD"],cwd=Path(__file__).parent,text=True).strip()
    print(json.dumps(dict(commit=rev,runs=rows),indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
