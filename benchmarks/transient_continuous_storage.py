"""Source-free decay with a smooth capacity, checked against field, mass and flux."""
import json
import math
from pathlib import Path
import subprocess

import numpy as np

from tensordg.advection_diffusion_1d import sample_solution, solve_transient_advection_diffusion_1d


def measure(speed, steps):
    # Reflection gives the reverse-flow solution with the same positive capacity.
    def oriented(x):
        return x if speed>0 else 1-x

    def profile(x):
        y = oriented(x)
        return 1+y+.2*y**2

    def storage(x):
        return (.9+.4*oriented(x))/profile(x)

    edges = np.linspace(0.,1.,11)**1.3
    times = np.linspace(0.,1.,steps+1)
    ends = profile(edges[[0,-1]])
    centers,states,_,(west,east) = solve_transient_advection_diffusion_1d(
        times,profile,degree=2,cell_edges=edges,speed=speed,peclet=4.,storage=storage,
        left_value=0.,right_value=0.,boundary_value_history=lambda t: ends*math.exp(-t),
        return_face_transport=True)
    field_error = 0.
    for t,state in zip(times,states):
        x,u = sample_solution(centers,state,cell_edges=edges)
        field_error = max(field_error,float((u-math.exp(-t)*profile(x)).abs().max()))
    # Use higher-order, independent quadrature for the numerical stored amount.
    q,w = np.polynomial.legendre.leggauss(24)
    widths = np.diff(edges)
    x = edges[:-1,None]+.5*widths[:,None]*(q+1)
    values = states.numpy() @ np.polynomial.legendre.legvander(q,2).T
    stored = .5*widths*((values*storage(x)) @ w)
    exact_stored = np.exp(-times[:,None])*widths*(.9+.4*oriented(edges[:-1]+.5*widths))
    current = speed*(profile(edges)-.25*(1+.4*oriented(edges)))
    integrated = current[None,:]*(-np.expm1(-np.diff(times))*np.exp(-times[:-1]))[:,None]
    flux_error = np.maximum(abs(west.numpy()-integrated[:,:-1]),abs(east.numpy()-integrated[:,1:]))
    return dict(field=field_error,cell_mass=float(abs(stored-exact_stored).max()),
                mean_flux=float((flux_error/np.diff(times)[:,None]).max()),
                balance=float(abs(np.diff(stored,axis=0)-(west-east).numpy()).max()))


if __name__ == "__main__":
    rows = []
    for speed in (-1.,1.):
        previous = {}
        for steps in (8,16,32,64):
            errors = measure(speed,steps)
            orders = {k:math.log2(previous[k]/errors[k]) for k in ("field","cell_mass","mean_flux") if previous}
            passed = errors["balance"]<1.e-10 and (steps<64 or min(orders.values())>1.85)
            rows.append(dict(speed=speed,steps=steps,errors=errors,orders=orders,passed=passed))
            previous = errors
    rev = subprocess.check_output(["git","rev-parse","HEAD"],cwd=Path(__file__).parent,text=True).strip()
    print(json.dumps(dict(commit=rev,runs=rows),indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
