"""Measure the fundamental decay rate of a two-layer column with absorbing ends."""
import json
import math
from pathlib import Path
import subprocess

import numpy as np
from scipy.linalg import expm
from scipy.optimize import brentq
import torch

from tensordg.advection_diffusion_1d import solve_transient_advection_diffusion_1d


def measure(speed):
    capacity = (1.2, 3.4)
    diffusion = (.25, .6)
    lengths = (.4, .6)

    def transfer(layer, rate, distance):
        d = diffusion[layer]
        matrix = np.array([[speed/d, -1/d], [rate*capacity[layer], 0.]])
        return expm(distance[...,None,None]*matrix)

    def mode_data(rate):
        initial = np.array([0., -1.])
        interface = transfer(0,rate,np.asarray(lengths[0])) @ initial
        end = transfer(1,rate,np.asarray(lengths[1])) @ interface
        return initial, interface, end

    # The first zero of the right-end concentration gives the fundamental mode.
    rates = np.linspace(0., 20., 201)
    for lo, hi in zip(rates,rates[1:]):
        if mode_data(hi)[2][0] <= 0.:
            break
    else:
        raise ValueError("fundamental eigenvalue was not bracketed")
    rate = brentq(lambda v:mode_data(v)[2][0],lo,hi)
    initial, interface, end = mode_data(rate)
    exact_mass = (end[1]-initial[1])/rate

    def profile(x):
        coords = x.numpy()
        left = coords<lengths[0]
        values = np.empty_like(coords)
        values[left] = (transfer(0,rate,coords[left]) @ initial)[...,0]
        values[~left] = (transfer(1,rate,coords[~left]-lengths[0]) @ interface)[...,0]
        return torch.from_numpy(values)

    rows = []
    previous = None
    for n in (4,8,16,32):
        edges = np.r_[np.linspace(0.,.4,n+1),np.linspace(.4,1.,n+1)[1:]]
        times = np.linspace(0.,.5,2*n+1)
        _,states,_ = solve_transient_advection_diffusion_1d(
            times,profile,cell_edges=edges,degree=3,speed=speed,peclet=4.,
            diffusivity_ratio=np.repeat(np.array(diffusion)*4,n),storage=np.repeat(capacity,n),
            left_value=0.,right_value=0.)
        stored = states[:,:,0].numpy() @ (np.diff(edges)*np.repeat(capacity,n))
        observed = -math.log(stored[-1]/stored[0])/times[-1]
        error = abs(observed-rate)
        order = math.log2(previous/error) if previous else None
        mass_error = float(abs(stored-exact_mass*np.exp(-rate*times)).max())
        rows.append(dict(speed=speed,cells_per_layer=n,reference_rate=rate,observed_rate=observed,
                         rate_error=error,mass_error=mass_error,order=order,
                         passed=bool(np.all(np.diff(stored)<0.) and (n<32 or
                                     (order>1.85 and error<.001 and mass_error<.0001)))))
        previous = error
    return rows


if __name__ == "__main__":
    rows = measure(-1.)+measure(1.)
    rev = subprocess.check_output(["git","rev-parse","HEAD"],cwd=Path(__file__).parent,text=True).strip()
    print(json.dumps(dict(commit=rev,runs=rows),indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
