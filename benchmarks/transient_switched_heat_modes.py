"""Off-grid switching between two insulated-plate Fourier heating modes."""
import json
import math
from pathlib import Path
import subprocess

import numpy as np

from tensordg import solve_transient_diffusion_2d


def measure(steps,damped):
    jumps = (.071,.163)
    end = .25
    times = np.linspace(0.,end,steps+1)
    kx,ky = .8,1.7
    def source(t,x,y):
        if t<jumps[0]:
            return 1.+2*(math.pi*x).cos()
        if t<jumps[1]:
            return .5-(math.pi*y).cos()
        return 0.
    bc = dict.fromkeys(("left","right","bottom","top"),"neumann")
    x,y,states = solve_transient_diffusion_2d(
        0.,times,ncells=(8,7),degree=3,bc=bc,conductivity=(kx,ky,0.),
        source_history=source,load_jump_times=jumps,rannacher=damped)
    def pulse_response(rate,start,stop):
        duration = np.clip(times-start,0.,stop-start)
        return -np.expm1(-rate*duration)*np.exp(-rate*np.maximum(times-stop,0.))/rate
    target_x = 2*pulse_response(kx*math.pi**2,0.,jumps[0])
    target_y = -pulse_response(ky*math.pi**2,*jumps)
    target_mean = np.minimum(times,jumps[0])+.5*np.clip(times-jumps[0],0.,jumps[1]-jumps[0])
    z,w = np.polynomial.legendre.leggauss(10)
    phi = np.polynomial.legendre.legvander(z,3)
    def moments(centers):
        coords = centers.numpy()[:,None]+z/(2*len(centers))
        return (np.cos(np.pi*coords)*w) @ phi/len(centers)
    # The cosine has squared L² norm 1/2 on the unit interval.
    modes_x = np.einsum("txyi,xi->t",states.numpy()[...,0],moments(x))/len(y)
    modes_y = np.einsum("txyj,yj->t",states.numpy()[:,:,:,0,:],moments(y))/len(x)
    means = states[:,:,:,0,0].mean(dim=(1,2)).numpy()
    return dict(x_mode=float(abs(modes_x-target_x).max()),y_mode=float(abs(modes_y-target_y).max()),
                mean=float(abs(means-target_mean).max()))


if __name__=="__main__":
    rows = []
    for damped in (False,True):
        previous = {}
        for steps in (8,16,32,64):
            errors = measure(steps,damped)
            orders = {k:math.log2(previous[k]/errors[k]) for k in ("x_mode","y_mode") if previous}
            passed = errors["mean"]<1e-11 and (steps<64 or min(orders.values())>1.65)
            rows.append(dict(steps=steps,rannacher=damped,errors=errors,orders=orders,passed=passed))
            previous = errors
    rev = subprocess.check_output(["git","rev-parse","HEAD"],cwd=Path(__file__).parent,text=True).strip()
    print(json.dumps(dict(commit=rev,runs=rows),indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
