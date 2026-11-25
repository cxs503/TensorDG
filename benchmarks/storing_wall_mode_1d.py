"""Spatial and temporal refinement of SIPG interval heat and storing-wall temperatures."""
import json
import math
from pathlib import Path
import subprocess

import numpy as np
import torch

from tensordg import solve_heat_1d_bulk_history


def measure(n, degree, substeps):
    k, capacity, beta, center = 1.4, 2., 1.8, .35
    wall_capacity = np.array([.3, .8])
    decay = k*beta**2/capacity
    faces = np.linspace(0., 1., n+1)
    phase = beta*(faces-center)
    face_flux = k*beta*np.sin(phase)
    wall_mode = np.cos(phase[[0,-1]])
    outward = face_flux[[0,-1]]*[-1., 1.]
    # Each surface cools into a zero-temperature ambient. Its film removes
    # the outward Fourier heat plus the heat released by the surface itself.
    films = outward/wall_mode+decay*wall_capacity
    knots = np.array([0., .07, .19, .3])
    dt = np.diff(knots)
    signal = np.exp(-decay*knots)
    integrated_signal = -signal[:-1]*np.expm1(-decay*dt)/decay
    _, modes, heat, bulk, wall, means = solve_heat_1d_bulk_history(
        knots, [(0.,0.)]*len(dt), initial_temperature=lambda x: torch.cos(beta*(x-center)),
        ncells=n, degree=degree, steps_per_interval=substeps, conductivity=k,
        capacity=capacity, heat_loss_coefficient=0., film_coefficients=films,
        wall_areal_heat_capacity=wall_capacity, return_wall_history=True, return_interval_modes=True)
    cell_mean = modes[:,:,0].numpy()
    exact_cell_mode = n*np.diff(np.sin(phase))/beta
    q = heat.numpy()
    surface = wall["temperature"].numpy()
    external = wall["external_heat"].numpy()
    # Evaluate the complete modal polynomial inside each cell, including both traces.
    local_points = np.linspace(-1., 1., 2*degree+3)
    positions = (faces[:-1,None]+(1+local_points)/(2*n))
    sampled = means.numpy() @ np.polynomial.legendre.legvander(local_points,degree).T
    reference = (integrated_signal/dt)[:,None,None]*np.cos(beta*(positions-center))
    errors = dict(interval_field_K=np.max(abs(sampled-reference)),
        cell_K=np.max(abs(cell_mean-signal[:,None]*exact_cell_mode)),
        wall_mean_K=np.max(abs(wall["mean_temperature"].numpy()-(integrated_signal/dt)[:,None]*wall_mode)),
        face_heat_J_m2=np.max(abs(q-integrated_signal[:,None]*face_flux)),
        external_heat_J_m2=np.max(abs(external+integrated_signal[:,None]*films*wall_mode)))
    cell_balance = capacity*np.diff(cell_mean,axis=0)/n+np.diff(q,axis=1)-bulk.numpy()
    wall_balance = wall_capacity*np.diff(surface,axis=0)+q[:,[0,-1]]*[1.,-1.]-external
    return errors, float(abs(cell_balance).max()), float(abs(wall_balance).max())


if __name__ == "__main__":
    rows = []
    for study in ("space","time"):
        previous = None
        for level in (1,2,4,8):
            n,degree,steps = (4*level,2,512) if study=="space" else (16,4,level)
            errors, cell, wall = measure(n,degree,steps)
            orders = {key:math.log2(previous[key]/value) for key,value in errors.items()} if previous else {}
            minimum_order = 2.6 if study=="space" else 1.9
            passed = max(cell,wall)<1e-10 and all(order>minimum_order for order in orders.values())
            rows.append(dict(passed=passed,study=study,ncells=n,degree=degree,steps_per_interval=steps,
                             errors=errors,orders=orders,cell_balance_J_m2=cell,wall_balance_J_m2=wall))
            previous = errors
    commit = subprocess.check_output(["git","rev-parse","HEAD"],cwd=Path(__file__).parent,text=True).strip()
    print(json.dumps(dict(commit=commit,runs=rows),indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
